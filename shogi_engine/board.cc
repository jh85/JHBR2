/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

#include "shogi/board.h"

#include <algorithm>
#include <cassert>
#include <sstream>

// =====================================================================
// Simple hash combiner (same approach as lc0's HashCat)
// =====================================================================

namespace {

uint64_t HashMix(uint64_t val) {
  return UINT64_C(0xfad0d7f2fbb059f1) * (val + UINT64_C(0xbaad41cdcb839961)) +
         UINT64_C(0x7acec0050bf82f43) * ((val >> 31) + UINT64_C(0xd571b3a92b1b2755));
}

uint64_t HashCombine(uint64_t hash, uint64_t x) {
  hash ^= UINT64_C(0x299799adf0d95def) + HashMix(x) + (hash << 6) + (hash >> 2);
  return hash;
}

}  // namespace

namespace lczero {

// =====================================================================
// Step attack tables (non-sliding pieces)
// =====================================================================

// Direction offsets for step attacks, indexed by [PieceType][Color].
// Each entry is a list of (file_delta, rank_delta) pairs terminated by {0,0}
// when the list is shorter than the max.
//
// Convention: positive rank = toward rank i (BLACK's forward = toward rank a
// = negative rank delta).  For WHITE, forward = positive rank delta.
//
// We compute these dynamically — no static tables needed for correctness.

Bitboard ShogiBoard::StepAttacks(PieceType pt, Color c, Square sq) {
  Bitboard bb = Bitboard::Zero();
  int f = sq.file().idx;
  int r = sq.rank().idx;

  // Helper: add square if in bounds.
  auto add = [&](int df, int dr) {
    int nf = f + df, nr = r + dr;
    if (nf >= 0 && nf < 9 && nr >= 0 && nr < 9) {
      bb.Set(Square(File::FromIdx(nf), Rank::FromIdx(nr)));
    }
  };

  // "Forward" depends on color: BLACK moves toward rank 0 (up), WHITE toward
  // rank 8 (down).
  int fwd = (c == BLACK) ? -1 : 1;

  if (pt == kPawn) {
    add(0, fwd);  // one step forward
  } else if (pt == kKnight) {
    // Shogi knight: 2 forward + 1 left/right (only forward, not backward)
    add(-1, 2 * fwd);
    add(+1, 2 * fwd);
  } else if (pt == kSilver) {
    add(0, fwd);      // forward
    add(-1, fwd);     // forward-right
    add(+1, fwd);     // forward-left
    add(-1, -fwd);    // backward-right
    add(+1, -fwd);    // backward-left
  } else if (pt == kGold || pt == kProPawn || pt == kProLance ||
             pt == kProKnight || pt == kProSilver) {
    // Gold and all promoted minor pieces move the same way.
    add(0, fwd);      // forward
    add(-1, fwd);     // forward-right
    add(+1, fwd);     // forward-left
    add(-1, 0);       // right
    add(+1, 0);       // left
    add(0, -fwd);     // backward
  } else if (pt == kKing) {
    for (int df = -1; df <= 1; ++df)
      for (int dr = -1; dr <= 1; ++dr)
        if (df || dr) add(df, dr);
  }
  // Horse and Dragon step attacks are handled separately (they also have
  // sliding components, so they go through PieceAttacks).
  // Horse extra steps: 4 cardinal directions (in addition to bishop slides)
  else if (pt == kHorse) {
    add(0, -1); add(0, +1); add(-1, 0); add(+1, 0);
  }
  // Dragon extra steps: 4 diagonal directions (in addition to rook slides)
  else if (pt == kDragon) {
    add(-1, -1); add(-1, +1); add(+1, -1); add(+1, +1);
  }
  // Lance has no step attacks (it's purely sliding).
  // Bishop and Rook have no step attacks (purely sliding).

  return bb;
}

// =====================================================================
// Sliding attacks (lance, bishop, rook, horse, dragon)
// =====================================================================

namespace {

// Slide in one direction until hitting a piece or the board edge.
Bitboard RayAttack(int f, int r, int df, int dr, const Bitboard& occ) {
  Bitboard bb = Bitboard::Zero();
  f += df;
  r += dr;
  while (f >= 0 && f < 9 && r >= 0 && r < 9) {
    Square sq(File::FromIdx(f), Rank::FromIdx(r));
    bb.Set(sq);
    if (occ.Test(sq)) break;  // Blocked.
    f += df;
    r += dr;
  }
  return bb;
}

}  // namespace

Bitboard ShogiBoard::SlidingAttacks(PieceType pt, Color c, Square sq,
                                    const Bitboard& occ) const {
  Bitboard bb = Bitboard::Zero();
  int f = sq.file().idx;
  int r = sq.rank().idx;
  int fwd = (c == BLACK) ? -1 : 1;

  if (pt == kLance) {
    bb |= RayAttack(f, r, 0, fwd, occ);
  } else if (pt == kBishop || pt == kHorse) {
    bb |= RayAttack(f, r, -1, -1, occ);
    bb |= RayAttack(f, r, -1, +1, occ);
    bb |= RayAttack(f, r, +1, -1, occ);
    bb |= RayAttack(f, r, +1, +1, occ);
  } else if (pt == kRook || pt == kDragon) {
    bb |= RayAttack(f, r, 0, -1, occ);
    bb |= RayAttack(f, r, 0, +1, occ);
    bb |= RayAttack(f, r, -1, 0, occ);
    bb |= RayAttack(f, r, +1, 0, occ);
  }

  return bb;
}

Bitboard ShogiBoard::PieceAttacks(PieceType pt, Color c, Square sq,
                                  const Bitboard& occ) const {
  Bitboard bb = StepAttacks(pt, c, sq);

  // Add sliding attacks for sliding pieces.
  if (pt == kLance || pt == kBishop || pt == kRook ||
      pt == kHorse || pt == kDragon) {
    bb |= SlidingAttacks(pt, c, sq, occ);
  }

  return bb;
}

// =====================================================================
// Attackers to a square
// =====================================================================

Bitboard ShogiBoard::AttackersTo(Square sq, const Bitboard& occ) const {
  Bitboard attackers = Bitboard::Zero();

  // For each color, check which pieces can reach this square.
  for (Color c : {BLACK, WHITE}) {
    // Step pieces: check if sq is in their attack set.
    // Trick: if a pawn at sq (as color ~c) attacks a square S,
    // then a pawn at S (as color c) can attack sq.
    // So we compute attacks FROM sq AS the defender, and intersect with
    // attacker pieces.

    // Pawn, Knight, Silver, Gold (and promoted equivalents), King
    PieceType step_types[] = {kPawn, kKnight, kSilver, kGold,
                              kProPawn, kProLance, kProKnight, kProSilver,
                              kKing};
    for (PieceType pt : step_types) {
      // Attacks from sq as if we were the OTHER color's piece of type pt
      // → these are the squares where an attacker of type pt/color c
      //   would need to be to attack sq.
      Bitboard reverse = StepAttacks(pt, ~c, sq);
      attackers |= (reverse & pieces(c, pt));
    }

    // Sliding pieces: lance, bishop, rook, horse, dragon.
    // For lance, direction depends on color.
    Bitboard lance_atk = SlidingAttacks(kLance, ~c, sq, occ);
    attackers |= (lance_atk & pieces(c, kLance));

    // Bishop/Horse share diagonal slides.
    Bitboard diag_atk = SlidingAttacks(kBishop, c, sq, occ);
    attackers |= (diag_atk & (pieces(c, kBishop) | pieces(c, kHorse)));

    // Rook/Dragon share straight slides.
    Bitboard straight_atk = SlidingAttacks(kRook, c, sq, occ);
    attackers |= (straight_atk & (pieces(c, kRook) | pieces(c, kDragon)));

    // Horse step attacks (cardinal directions).
    Bitboard horse_step = StepAttacks(kHorse, ~c, sq);
    attackers |= (horse_step & pieces(c, kHorse));

    // Dragon step attacks (diagonal directions).
    Bitboard dragon_step = StepAttacks(kDragon, ~c, sq);
    attackers |= (dragon_step & pieces(c, kDragon));
  }

  return attackers;
}

bool ShogiBoard::InCheck(Color c) const {
  Bitboard atk = AttackersTo(king_sq_[c]);
  return (atk & pieces(~c)).Any();
}

// =====================================================================
// Board manipulation
// =====================================================================

void ShogiBoard::PutPiece(Square sq, Piece pc) {
  assert(board_[sq.as_idx()].IsNone());
  board_[sq.as_idx()] = pc;
  by_color_[pc.GetColor()].Set(sq);
  by_type_[pc.GetType().idx].Set(sq);
  if (pc.GetType() == kKing) {
    king_sq_[pc.GetColor()] = sq;
  }
}

Piece ShogiBoard::RemovePiece(Square sq) {
  Piece pc = board_[sq.as_idx()];
  assert(!pc.IsNone());
  board_[sq.as_idx()] = Piece::None();
  by_color_[pc.GetColor()].Clear(sq);
  by_type_[pc.GetType().idx].Clear(sq);
  return pc;
}

void ShogiBoard::MovePiece(Square from, Square to) {
  Piece pc = RemovePiece(from);
  PutPiece(to, pc);
}

// =====================================================================
// Move application
// =====================================================================

UndoInfo ShogiBoard::DoMove(Move m) {
  UndoInfo undo;
  Color us = side_to_move_;
  undo.prev_hand = hand_[us];
  undo.prev_hash = hash_;
  undo.prev_continuous_check = continuous_check_[us];

  // Save current position to history (before making the move).
  history_.push_back({hash_, hand_[BLACK].raw(), hand_[WHITE].raw()});

  if (m.is_drop()) {
    // Drop: remove from hand, place on board.
    PieceType pt = m.drop_piece();
    undo.captured = Piece::None();
    hand_[us].Sub(pt);
    PutPiece(m.to(), Piece::Make(us, pt));
  } else {
    // Board move: handle capture, then move piece.
    Square to = m.to();
    Square from = m.from();

    // Capture?
    if (!empty(to)) {
      Piece captured = RemovePiece(to);
      undo.captured = captured;
      // Add captured piece to hand (unpromoted form).
      PieceType cap_base = captured.GetType().Unpromote();
      hand_[us].Add(cap_base);
    } else {
      undo.captured = Piece::None();
    }

    // Move the piece.
    Piece moved = RemovePiece(from);

    // Promote if flagged.
    if (m.is_promotion()) {
      moved = moved.Promote();
    }

    PutPiece(to, moved);
  }

  side_to_move_ = ~us;
  ply_++;

  // Recompute hash for the new position.
  ComputeHash();

  // Update continuous check counter.
  // If the opponent's king is now in check, increment our counter.
  // Otherwise, reset it.
  if (InCheck(~us)) {
    continuous_check_[us] += 1;
  } else {
    continuous_check_[us] = 0;
  }

  return undo;
}

void ShogiBoard::UndoMove(Move m, const UndoInfo& undo) {
  side_to_move_ = ~side_to_move_;
  ply_--;
  Color us = side_to_move_;

  if (m.is_drop()) {
    // Un-drop: remove from board, add back to hand.
    RemovePiece(m.to());
    hand_[us] = undo.prev_hand;
  } else {
    Square to = m.to();
    Square from = m.from();

    // Remove moved piece from destination.
    Piece moved = RemovePiece(to);

    // Unpromote if the move was a promotion.
    if (m.is_promotion()) {
      moved = moved.Unpromote();
    }

    // Put it back at the source.
    PutPiece(from, moved);

    // Restore captured piece.
    if (!undo.captured.IsNone()) {
      PutPiece(to, undo.captured);
      // Restore hand.
      hand_[us] = undo.prev_hand;
    }
  }

  // Restore hash and continuous check counter.
  hash_ = undo.prev_hash;
  continuous_check_[us] = undo.prev_continuous_check;

  // Remove the history entry we added in DoMove.
  if (!history_.empty()) {
    history_.pop_back();
  }
}

// =====================================================================
// Legal move generation
// =====================================================================

void ShogiBoard::GenerateBoardMoves(MoveList& moves) const {
  Color us = side_to_move_;
  Bitboard our = pieces(us);
  Bitboard occ = occupied();
  Bitboard their = pieces(~us);

  Bitboard tmp = our;
  while (tmp.Any()) {
    Square from = tmp.Pop();
    Piece pc = piece_on(from);
    PieceType pt = pc.GetType();

    // Get attack squares for this piece.
    Bitboard targets = PieceAttacks(pt, us, from, occ);

    // Can't capture own pieces.
    targets &= ~our;

    targets.ForEach([&](Square to) {
      bool can_promote = false;
      bool must_promote = false;

      if (pt.CanPromote()) {
        // Can promote if from or to is in promotion zone.
        can_promote = from.InPromotionZone(us) || to.InPromotionZone(us);

        // Must promote if the piece can't move further from the dest:
        Rank dest_rank = to.rank();
        Rank rel_rank = (us == BLACK) ? dest_rank
                                      : Rank::FromIdx(8 - dest_rank.idx);
        if (pt == kPawn || pt == kLance) {
          must_promote = (rel_rank.idx == 0);  // rank a for BLACK
        } else if (pt == kKnight) {
          must_promote = (rel_rank.idx <= 1);  // ranks a,b for BLACK
        }
      }

      if (can_promote) {
        moves.push_back(Move::Promotion(from, to));
      }
      if (!must_promote) {
        moves.push_back(Move::Normal(from, to));
      }
    });
  }
}

void ShogiBoard::GenerateDropMoves(MoveList& moves) const {
  Color us = side_to_move_;
  Bitboard empty_sq = ~occupied();

  // Ensure we only iterate valid squares (mask out non-board bits).
  empty_sq &= Bitboard::All();

  // Pawn column restriction: can't have two unpromoted pawns on same file.
  Bitboard our_pawns = pieces(us, kPawn);

  for (int pt_idx = kPawn.idx; pt_idx <= kGold.idx; ++pt_idx) {
    PieceType pt = PieceType::FromIdx(pt_idx);
    if (!hand_[us].Has(pt)) continue;

    Bitboard targets = empty_sq;

    // Rank restrictions: pieces that can only move forward can't be
    // dropped where they'd have no legal moves.
    if (pt == kPawn || pt == kLance) {
      // BLACK can't drop on rank a (idx 0); WHITE can't drop on rank i (idx 8)
      Rank forbidden = (us == BLACK) ? kRank1 : kRank9;
      targets &= ~ShogiTables::RankBB[forbidden.idx];
    }
    if (pt == kKnight) {
      // BLACK can't drop on ranks a,b; WHITE can't drop on ranks h,i
      if (us == BLACK) {
        targets &= ~ShogiTables::RankBB[0];
        targets &= ~ShogiTables::RankBB[1];
      } else {
        targets &= ~ShogiTables::RankBB[7];
        targets &= ~ShogiTables::RankBB[8];
      }
    }

    // Pawn: can't drop on a file that already has an unpromoted pawn (二歩).
    if (pt == kPawn) {
      for (int f = 0; f < 9; ++f) {
        if ((our_pawns & ShogiTables::FileBB[f]).Any()) {
          targets &= ~ShogiTables::FileBB[f];
        }
      }
    }

    targets.ForEach([&](Square to) {
      moves.push_back(Move::Drop(pt, to));
    });
  }
}

MoveList ShogiBoard::GenerateLegalMoves() const {
  MoveList pseudo;
  pseudo.reserve(128);

  GenerateBoardMoves(pseudo);
  GenerateDropMoves(pseudo);

  // Filter: only keep moves that don't leave our king in check.
  MoveList legal;
  legal.reserve(pseudo.size());

  for (const Move& m : pseudo) {
    if (IsLegal(m)) {
      legal.push_back(m);
    }
  }

  return legal;
}

bool ShogiBoard::IsLegal(Move m) const {
  // Apply the move, check if our king is in check, undo.
  ShogiBoard copy = *this;
  UndoInfo undo = copy.DoMove(m);

  // After DoMove, side_to_move_ has flipped. Our king must not be in check.
  Color us = side_to_move_;  // The side that just moved (before flip in copy).
  bool legal = !copy.InCheck(us);

  // Additional check for pawn drop mate (打ち歩詰め):
  // You can't drop a pawn that gives checkmate.
  if (legal && m.is_drop() && m.drop_piece() == kPawn) {
    if (copy.InCheck(~us)) {
      // The pawn drop gives check. Check if it's checkmate.
      MoveList responses = copy.GenerateLegalMoves();
      if (responses.empty()) {
        legal = false;  // Pawn drop checkmate is illegal.
      }
    }
  }

  return legal;
}

// =====================================================================
// Game result
// =====================================================================

ShogiBoard::GameResult ShogiBoard::ComputeGameResult() const {
  // Check for declaration win first.
  if (CanDeclareWin()) {
    return GameResult::kDeclarationWin;
  }
  // In Shogi, if the side to move has no legal moves, it's checkmate.
  // (There is no stalemate — no legal moves = loss.)
  MoveList moves = GenerateLegalMoves();
  if (moves.empty()) {
    return GameResult::kCheckmate;
  }
  return GameResult::kUndecided;
}

// =====================================================================
// Entering-king declaration (入玉宣言)
// =====================================================================

ShogiBoard::EnteringKingInfo ShogiBoard::ComputeEnteringKingInfo(Color c) const {
  EnteringKingInfo info = {0, 0, false};

  // Enemy camp: last 3 ranks from the given color's perspective.
  // BLACK's enemy camp = ranks 0,1,2 (top).  WHITE's = ranks 6,7,8 (bottom).
  Bitboard enemy_camp = ShogiTables::PromotionZoneBB[c];

  // Is king in enemy camp?
  info.king_in_camp = enemy_camp.Test(king_sq_[c]);

  // Count our pieces in enemy camp (excluding king).
  Bitboard our_in_camp = pieces(c) & enemy_camp;
  int total_in_camp = our_in_camp.PopCount();
  if (info.king_in_camp) total_in_camp--;  // Exclude king
  info.pieces_in_camp = total_in_camp;

  // Count points.
  // Major pieces (R, B, Dragon, Horse) in enemy camp = 5 pts each.
  // Other pieces in enemy camp = 1 pt each.
  Bitboard major_in_camp = our_in_camp &
      (by_type_[kBishop.idx] | by_type_[kRook.idx] |
       by_type_[kHorse.idx] | by_type_[kDragon.idx]);
  int major_count = major_in_camp.PopCount();
  int minor_count = info.pieces_in_camp - major_count;

  int points = major_count * 5 + minor_count;

  // Add hand pieces.
  Hand h = hand_[c];
  // Minor hand pieces: P, L, N, S, G = 1 pt each.
  points += h.Count(kPawn) + h.Count(kLance) + h.Count(kKnight)
          + h.Count(kSilver) + h.Count(kGold);
  // Major hand pieces: B, R = 5 pts each.
  points += (h.Count(kBishop) + h.Count(kRook)) * 5;

  info.points = points;
  return info;
}

bool ShogiBoard::CanDeclareWin() const {
  Color us = side_to_move_;

  // (5) King must not be in check.
  if (InCheck(us)) return false;

  // Compute entering king info.
  EnteringKingInfo info = ComputeEnteringKingInfo(us);

  // (2) King must be in enemy camp.
  if (!info.king_in_camp) return false;

  // (4) At least 10 pieces (excluding king) in enemy camp.
  if (info.pieces_in_camp < 10) return false;

  // (3) Point threshold: BLACK needs 28+, WHITE needs 27+.
  int threshold = (us == BLACK) ? 28 : 27;
  if (info.points < threshold) return false;

  return true;
}

// =====================================================================
// SFEN parsing
// =====================================================================

ShogiBoard::ShogiBoard() {
  board_.fill(Piece::None());
  for (auto& bb : by_type_) bb = Bitboard::Zero();
  by_color_[BLACK] = Bitboard::Zero();
  by_color_[WHITE] = Bitboard::Zero();
  hand_[BLACK] = Hand();
  hand_[WHITE] = Hand();
  king_sq_[BLACK] = kSquareNone;
  king_sq_[WHITE] = kSquareNone;
}

void ShogiBoard::SetStartPos() { SetFromSfen(kStartingSfen); }

bool ShogiBoard::SetFromSfen(const std::string& sfen) {
  // Clear.
  *this = ShogiBoard();

  std::istringstream ss(sfen);
  std::string board_str, side_str, hand_str, ply_str;
  ss >> board_str >> side_str >> hand_str >> ply_str;

  // 1. Board placement: ranks separated by '/', files left to right = 9..1.
  // SFEN board goes top-to-bottom (rank a first), left-to-right (file 9 first).
  int f = 8, r = 0;  // Start at file 9 (idx 8), rank a (idx 0).
  bool promoted = false;

  for (char ch : board_str) {
    if (ch == '/') {
      f = 8;
      r++;
      continue;
    }
    if (ch == '+') {
      promoted = true;
      continue;
    }
    if (ch >= '1' && ch <= '9') {
      f -= (ch - '0');
      continue;
    }

    // Piece character.
    Color c = (ch >= 'A' && ch <= 'Z') ? BLACK : WHITE;
    PieceType pt = PieceType::Parse(ch);
    if (!pt.IsValid()) return false;
    if (promoted) {
      pt = pt.Promote();
      promoted = false;
    }

    Square sq(File::FromIdx(f), Rank::FromIdx(r));
    PutPiece(sq, Piece::Make(c, pt));
    f--;
  }

  // 2. Side to move.
  side_to_move_ = (side_str == "w") ? WHITE : BLACK;

  // 3. Hand pieces.
  if (hand_str != "-") {
    int count = 0;
    for (char ch : hand_str) {
      if (ch >= '0' && ch <= '9') {
        count = count * 10 + (ch - '0');
        continue;
      }
      if (count == 0) count = 1;
      Color c = (ch >= 'A' && ch <= 'Z') ? BLACK : WHITE;
      PieceType pt = PieceType::Parse(ch);
      for (int i = 0; i < count; ++i) {
        hand_[c].Add(pt);
      }
      count = 0;
    }
  }

  // 4. Ply.
  try {
    ply_ = ply_str.empty() ? 1 : std::stoi(ply_str);
  } catch (...) {
    ply_ = 1;
  }

  // Compute initial hash and clear history.
  ComputeHash();
  ClearHistory();

  return true;
}

std::string ShogiBoard::ToSfen() const {
  std::string s;

  // 1. Board.
  for (int r = 0; r < 9; ++r) {
    if (r > 0) s += '/';
    int empty_count = 0;
    for (int f = 8; f >= 0; --f) {
      Square sq(File::FromIdx(f), Rank::FromIdx(r));
      Piece pc = piece_on(sq);
      if (pc.IsNone()) {
        empty_count++;
      } else {
        if (empty_count > 0) {
          s += std::to_string(empty_count);
          empty_count = 0;
        }
        PieceType pt = pc.GetType();
        if (pt.IsPromoted()) {
          s += '+';
          pt = pt.Unpromote();
        }
        char c = pt.ToChar();
        if (pc.GetColor() == WHITE) c = std::tolower(c);
        s += c;
      }
    }
    if (empty_count > 0) s += std::to_string(empty_count);
  }

  // 2. Side to move.
  s += (side_to_move_ == BLACK) ? " b " : " w ";

  // 3. Hand pieces.
  std::string hand_str;
  for (Color c : {BLACK, WHITE}) {
    PieceType pts[] = {kRook, kBishop, kGold, kSilver, kKnight, kLance, kPawn};
    for (PieceType pt : pts) {
      int cnt = hand_[c].Count(pt);
      if (cnt == 0) continue;
      if (cnt > 1) hand_str += std::to_string(cnt);
      char ch = pt.ToChar();
      if (c == WHITE) ch = std::tolower(ch);
      hand_str += ch;
    }
  }
  s += hand_str.empty() ? "-" : hand_str;

  // 4. Ply.
  s += " " + std::to_string(ply_);

  return s;
}

// =====================================================================
// Position hashing
// =====================================================================

void ShogiBoard::ComputeHash() {
  uint64_t h = 0;
  // Hash board pieces.
  for (int sq = 0; sq < kSquareNB; ++sq) {
    if (!board_[sq].IsNone()) {
      h = HashCombine(h, sq * 32 + board_[sq].val);
    }
  }
  // Hash hand pieces.
  h = HashCombine(h, hand_[BLACK].raw());
  h = HashCombine(h, hand_[WHITE].raw() + UINT64_C(0x1234567890));
  // Hash side to move.
  if (side_to_move_ == WHITE) {
    h = HashCombine(h, UINT64_C(0xABCDEF0123456789));
  }
  hash_ = h;
}

// =====================================================================
// Sennichite (repetition) detection
// =====================================================================

void ShogiBoard::ClearHistory() {
  history_.clear();
  continuous_check_[BLACK] = 0;
  continuous_check_[WHITE] = 0;
}

int ShogiBoard::RepetitionCount() const {
  int count = 0;
  for (const auto& entry : history_) {
    if (entry.hash == hash_ &&
        entry.hand_black == hand_[BLACK].raw() &&
        entry.hand_white == hand_[WHITE].raw()) {
      count++;
    }
  }
  return count;
}

ShogiBoard::RepetitionResult ShogiBoard::CheckRepetition() const {
  // Count how many times the current position has appeared in history.
  // Sennichite occurs on the 4th occurrence (3 previous + current = 4).

  int occurrences = 0;
  int last_match_distance = 0;

  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
    const auto& entry = history_[i];
    if (entry.hash == hash_ &&
        entry.hand_black == hand_[BLACK].raw() &&
        entry.hand_white == hand_[WHITE].raw()) {
      occurrences++;
      if (occurrences == 1) {
        last_match_distance = static_cast<int>(history_.size()) - i;
      }
      if (occurrences >= 3) {
        // 4th occurrence (3 in history + current) → sennichite.
        // Check for perpetual check.
        // If the side to move has been continuously checking for at least
        // as many plies as the distance to the first repetition,
        // that side LOSES (perpetual check).
        // If the opponent was continuously checking, we WIN.

        Color us = side_to_move_;
        int dist = static_cast<int>(history_.size()) - i;

        if (continuous_check_[us] >= dist) {
          return RepetitionResult::kLoss;  // We were giving perpetual check
        }
        if (continuous_check_[~us] >= dist) {
          return RepetitionResult::kWin;   // Opponent was giving perpetual check
        }
        return RepetitionResult::kDraw;
      }
    }
  }

  return RepetitionResult::kNone;
}

// =====================================================================
// Perspective flip (180° rotation + color swap)
// =====================================================================

ShogiBoard ShogiBoard::Flipped() const {
  ShogiBoard flipped;
  flipped.side_to_move_ = ~side_to_move_;
  flipped.ply_ = ply_;

  // Swap hands.
  flipped.hand_[BLACK] = hand_[WHITE];
  flipped.hand_[WHITE] = hand_[BLACK];

  // Rotate all pieces.
  for (int sq = 0; sq < kSquareNB; ++sq) {
    Piece pc = board_[sq];
    if (!pc.IsNone()) {
      Square flipped_sq = Square::FromIdx(80 - sq);
      Color flipped_color = ~pc.GetColor();
      flipped.PutPiece(flipped_sq, Piece::Make(flipped_color, pc.GetType()));
    }
  }

  return flipped;
}

// =====================================================================
// Debug string
// =====================================================================

std::string ShogiBoard::DebugString() const {
  std::string s;
  s += "  9 8 7 6 5 4 3 2 1\n";
  s += "+-------------------+\n";
  for (int r = 0; r < 9; ++r) {
    s += "|";
    for (int f = 8; f >= 0; --f) {
      Square sq(File::FromIdx(f), Rank::FromIdx(r));
      Piece pc = piece_on(sq);
      if (pc.IsNone()) {
        s += " .";
      } else {
        PieceType pt = pc.GetType();
        char c = pt.ToChar();
        if (pc.GetColor() == WHITE) c = std::tolower(c);
        if (pt.IsPromoted()) {
          s += "+";
          c = pt.Unpromote().ToChar();
          if (pc.GetColor() == WHITE) c = std::tolower(c);
        } else {
          s += " ";
        }
        s += c;
      }
    }
    s += "| ";
    s += char('a' + r);
    s += '\n';
  }
  s += "+-------------------+\n";

  // Hand pieces.
  for (Color c : {BLACK, WHITE}) {
    s += (c == BLACK) ? "Black hand: " : "White hand: ";
    PieceType pts[] = {kRook, kBishop, kGold, kSilver, kKnight, kLance, kPawn};
    bool any = false;
    for (PieceType pt : pts) {
      int cnt = hand_[c].Count(pt);
      if (cnt > 0) {
        if (cnt > 1) s += std::to_string(cnt);
        s += pt.ToChar();
        any = true;
      }
    }
    if (!any) s += "-";
    s += "\n";
  }

  s += std::string("Side to move: ") + (side_to_move_ == BLACK ? "BLACK" : "WHITE") + "\n";
  return s;
}

}  // namespace lczero
