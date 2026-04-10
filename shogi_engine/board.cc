/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

#include "shogi/board.h"

#include <algorithm>
#include <cassert>
#include <sstream>

namespace lczero {

// =====================================================================
// Zobrist hash tables
// =====================================================================

namespace Zobrist {

// Random keys for each piece on each square.
// Index: [piece.val][square] where piece.val covers BLACK (1-14) and WHITE (17-30).
static uint64_t Psq[32][kSquareNB];

// Random keys for hand piece counts.
// Index: [color][piece_type_idx][count] (max count: pawn=18, others≤4).
static uint64_t HandPiece[COLOR_NB][8][20];

// XOR this when side to move is WHITE.
static uint64_t Side;

// Simple PRNG for table initialization (splitmix64).
static uint64_t s_seed = UINT64_C(0x70736575646F7267);

static uint64_t Rand64() {
  s_seed += UINT64_C(0x9E3779B97F4A7C15);
  uint64_t z = s_seed;
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

static bool s_initialized = false;

void Init() {
  if (s_initialized) return;
  s_initialized = true;

  for (int pc = 0; pc < 32; ++pc)
    for (int sq = 0; sq < kSquareNB; ++sq)
      Psq[pc][sq] = Rand64();

  for (int c = 0; c < COLOR_NB; ++c)
    for (int pt = 0; pt < 8; ++pt)
      for (int n = 0; n < 20; ++n)
        HandPiece[c][pt][n] = Rand64();

  Side = Rand64();
}

}  // namespace Zobrist

// Hash a hand by XORing in the Zobrist key for each piece's count.
static uint64_t HashHand(Color c, const Hand& h) {
  uint64_t z = 0;
  for (int pt = kPawn.idx; pt <= kGold.idx; ++pt) {
    int n = h.Count(PieceType::FromIdx(pt));
    if (n > 0) z ^= Zobrist::HandPiece[c][pt][n];
  }
  return z;
}

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
  int i = sq.as_idx();
  if (pt == kPawn)    return ShogiTables::PawnEffectBB[i][c];
  if (pt == kKnight)  return ShogiTables::KnightEffectBB[i][c];
  if (pt == kSilver)  return ShogiTables::SilverEffectBB[i][c];
  if (pt == kGold || pt == kProPawn || pt == kProLance ||
      pt == kProKnight || pt == kProSilver)
                      return ShogiTables::GoldEffectBB[i][c];
  if (pt == kKing)    return ShogiTables::KingEffectBB[i];
  if (pt == kHorse)   return ShogiTables::HorseStepBB[i];
  if (pt == kDragon)  return ShogiTables::DragonStepBB[i];
  return Bitboard::Zero();
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
  if (pt == kLance) {
    return ShogiTables::LanceEffect(c, sq, occ);
  }
  if (pt == kBishop || pt == kHorse) {
    return ShogiTables::BishopEffect(sq, occ);
  }
  if (pt == kRook || pt == kDragon) {
    return ShogiTables::RookFileEffect(sq, occ) |
           ShogiTables::RookRankEffect(sq, occ);
  }
  return Bitboard::Zero();
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
  int i = sq.as_idx();
  Bitboard attackers = Bitboard::Zero();

  // Step attacks: reverse-perspective lookup.
  for (Color c : {BLACK, WHITE}) {
    attackers |= ShogiTables::PawnEffectBB[i][~c] & pieces(c, kPawn);
    attackers |= ShogiTables::KnightEffectBB[i][~c] & pieces(c, kKnight);
    attackers |= ShogiTables::SilverEffectBB[i][~c] & pieces(c, kSilver);
    attackers |= ShogiTables::GoldEffectBB[i][~c] &
        (pieces(c, kGold) | pieces(c, kProPawn) | pieces(c, kProLance) |
         pieces(c, kProKnight) | pieces(c, kProSilver));

    // Lance: use fast Qugiy effect (direction depends on color).
    attackers |= ShogiTables::LanceEffect(~c, sq, occ) & pieces(c, kLance);
  }

  // King.
  attackers |= ShogiTables::KingEffectBB[i] & by_type_[kKing.idx];

  // Rook/Dragon: full sliding (vertical + horizontal, both Qugiy).
  Bitboard straight = ShogiTables::RookFileEffect(sq, occ) |
                      ShogiTables::RookRankEffect(sq, occ);
  attackers |= straight & (by_type_[kRook.idx] | by_type_[kDragon.idx]);

  // Bishop/Horse: diagonal sliding (Qugiy).
  Bitboard diag = ShogiTables::BishopEffect(sq, occ);
  attackers |= diag & (by_type_[kBishop.idx] | by_type_[kHorse.idx]);

  // Horse/Dragon step attacks.
  attackers |= ShogiTables::HorseStepBB[i] & by_type_[kHorse.idx];
  attackers |= ShogiTables::DragonStepBB[i] & by_type_[kDragon.idx];

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

    // Hash: remove old hand state, update hand, add new hand state.
    hash_ ^= HashHand(us, hand_[us]);
    hand_[us].Sub(pt);
    hash_ ^= HashHand(us, hand_[us]);

    // Place piece on board.
    Piece pc = Piece::Make(us, pt);
    PutPiece(m.to(), pc);
    hash_ ^= Zobrist::Psq[pc.val][m.to().as_idx()];
  } else {
    Square to = m.to();
    Square from = m.from();

    // Capture?
    if (!empty(to)) {
      Piece captured = piece_on(to);
      undo.captured = captured;

      // Remove captured piece from hash and board.
      hash_ ^= Zobrist::Psq[captured.val][to.as_idx()];
      RemovePiece(to);

      // Add captured piece (unpromoted) to hand.
      PieceType cap_base = captured.GetType().Unpromote();
      hash_ ^= HashHand(us, hand_[us]);
      hand_[us].Add(cap_base);
      hash_ ^= HashHand(us, hand_[us]);
    } else {
      undo.captured = Piece::None();
    }

    // Remove moving piece from source.
    Piece moved = piece_on(from);
    hash_ ^= Zobrist::Psq[moved.val][from.as_idx()];
    RemovePiece(from);

    // Promote if flagged.
    if (m.is_promotion()) {
      moved = moved.Promote();
    }

    // Place at destination.
    PutPiece(to, moved);
    hash_ ^= Zobrist::Psq[moved.val][to.as_idx()];
  }

  // Flip side to move.
  hash_ ^= Zobrist::Side;
  side_to_move_ = ~us;
  ply_++;

  // Update continuous check counter.
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

MoveList ShogiBoard::GenerateLegalMoves() {
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

bool ShogiBoard::IsLegal(Move m) {
  // Apply the move in-place, check if our king is in check, then undo.
  Color us = side_to_move_;
  UndoInfo undo = DoMove(m);

  bool legal = !InCheck(us);

  // Additional check for pawn drop mate (打ち歩詰め):
  // You can't drop a pawn that gives checkmate.
  if (legal && m.is_drop() && m.drop_piece() == kPawn) {
    if (InCheck(~us)) {
      // The pawn drop gives check. Check if it's checkmate.
      MoveList responses = GenerateLegalMoves();
      if (responses.empty()) {
        legal = false;
      }
    }
  }

  UndoMove(m, undo);
  return legal;
}

// =====================================================================
// Game result
// =====================================================================

ShogiBoard::GameResult ShogiBoard::ComputeGameResult() {
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
  Zobrist::Init();
  uint64_t h = 0;
  for (int sq = 0; sq < kSquareNB; ++sq) {
    if (!board_[sq].IsNone()) {
      h ^= Zobrist::Psq[board_[sq].val][sq];
    }
  }
  h ^= HashHand(BLACK, hand_[BLACK]);
  h ^= HashHand(WHITE, hand_[WHITE]);
  if (side_to_move_ == WHITE) {
    h ^= Zobrist::Side;
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
