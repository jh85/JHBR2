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

Bitboard ShogiBoard::SlidingAttacks(PieceType pt, Color c, Square sq,
                                    const Bitboard& occ) const {
  if (pt == kLance) {
    return ShogiTables::LanceEffect(c, sq, occ);
  }
  if (pt == kBishop || pt == kHorse) {
    return ShogiTables::BishopEffect(sq, occ);
  }
  if (pt == kRook || pt == kDragon) {
    return ShogiTables::RookEffect(sq, occ);
  }
  return Bitboard::Zero();
}

Bitboard ShogiBoard::PieceAttacks(PieceType pt, Color c, Square sq,
                                  const Bitboard& occ) const {
  const int i = sq.as_idx();
  switch (pt.idx) {
    case kPawn.idx:
      return ShogiTables::PawnEffectBB[i][c];
    case kLance.idx:
      return ShogiTables::LanceEffect(c, sq, occ);
    case kKnight.idx:
      return ShogiTables::KnightEffectBB[i][c];
    case kSilver.idx:
      return ShogiTables::SilverEffectBB[i][c];
    case kBishop.idx:
      return ShogiTables::BishopEffect(sq, occ);
    case kRook.idx:
      return ShogiTables::RookEffect(sq, occ);
    case kGold.idx:
    case kProPawn.idx:
    case kProLance.idx:
    case kProKnight.idx:
    case kProSilver.idx:
      return ShogiTables::GoldEffectBB[i][c];
    case kKing.idx:
      return ShogiTables::KingEffectBB[i];
    case kHorse.idx:
      return ShogiTables::BishopEffect(sq, occ) | ShogiTables::HorseStepBB[i];
    case kDragon.idx:
      return ShogiTables::RookEffect(sq, occ) | ShogiTables::DragonStepBB[i];
    default:
      return Bitboard::Zero();
  }
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
  Bitboard straight = ShogiTables::RookEffect(sq, occ);
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
  return IsSquareAttacked(king_sq_[c], occupied(), ~c);
}

bool ShogiBoard::IsSquareAttacked(Square sq, const Bitboard& occ,
                                  Color attacker) const {
  int i = sq.as_idx();

  if ((ShogiTables::PawnEffectBB[i][~attacker] & pieces(attacker, kPawn)).Any())
    return true;
  if ((ShogiTables::KnightEffectBB[i][~attacker] & pieces(attacker, kKnight)).Any())
    return true;
  if ((ShogiTables::SilverEffectBB[i][~attacker] & pieces(attacker, kSilver)).Any())
    return true;

  Bitboard golds = pieces(attacker, kGold) | pieces(attacker, kProPawn) |
                   pieces(attacker, kProLance) | pieces(attacker, kProKnight) |
                   pieces(attacker, kProSilver);
  if ((ShogiTables::GoldEffectBB[i][~attacker] & golds).Any())
    return true;

  if ((ShogiTables::KingEffectBB[i] & pieces(attacker, kKing)).Any())
    return true;
  if ((ShogiTables::HorseStepBB[i] & pieces(attacker, kHorse)).Any())
    return true;
  if ((ShogiTables::DragonStepBB[i] & pieces(attacker, kDragon)).Any())
    return true;

  Bitboard lances = pieces(attacker, kLance);
  if ((ShogiTables::LanceMaskBB[i][~attacker] & lances).Any()) {
    if ((ShogiTables::LanceEffect(~attacker, sq, occ) & lances).Any())
      return true;
  }

  Bitboard rook_like = pieces(attacker, kRook) | pieces(attacker, kDragon);
  if ((ShogiTables::RookEffectBB[i] & rook_like).Any()) {
    if ((ShogiTables::RookEffect(sq, occ) & rook_like).Any())
      return true;
  }

  Bitboard bishop_like = pieces(attacker, kBishop) | pieces(attacker, kHorse);
  if ((ShogiTables::BishopEffectBB[i] & bishop_like).Any()) {
    if ((ShogiTables::BishopEffect(sq, occ) & bishop_like).Any())
      return true;
  }

  return false;
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
  return DoMoveInternal(m, true, -1);
}

UndoInfo ShogiBoard::DoMove(Move m, bool gives_check) {
  return DoMoveInternal(m, true, gives_check ? 1 : 0);
}

UndoInfo ShogiBoard::DoMoveInternal(Move m, bool update_auxiliary,
                                    int gives_check) {
  UndoInfo undo;
  Color us = side_to_move_;
  undo.prev_hand = hand_[us];
  undo.prev_hash = hash_;
  undo.prev_continuous_check = continuous_check_[us];
  undo.updated_auxiliary = update_auxiliary;

  // Save current position to history (before making the move).
  if (update_auxiliary) {
    history_.push_back({hash_, hand_[BLACK].raw(), hand_[WHITE].raw()});
  }

  if (m.is_drop()) {
    // Drop: remove from hand, place on board.
    PieceType pt = m.drop_piece();
    undo.captured = Piece::None();

    // Hash: remove old hand state, update hand, add new hand state.
    if (update_auxiliary) hash_ ^= HashHand(us, hand_[us]);
    hand_[us].Sub(pt);
    if (update_auxiliary) hash_ ^= HashHand(us, hand_[us]);

    // Place piece on board.
    Piece pc = Piece::Make(us, pt);
    PutPiece(m.to(), pc);
    if (update_auxiliary) hash_ ^= Zobrist::Psq[pc.val][m.to().as_idx()];
  } else {
    Square to = m.to();
    Square from = m.from();

    // Capture?
    if (!empty(to)) {
      Piece captured = piece_on(to);
      undo.captured = captured;

      // Remove captured piece from hash and board.
      if (update_auxiliary) hash_ ^= Zobrist::Psq[captured.val][to.as_idx()];
      RemovePiece(to);

      // Add captured piece (unpromoted) to hand.
      PieceType cap_base = captured.GetType().Unpromote();
      if (update_auxiliary) hash_ ^= HashHand(us, hand_[us]);
      hand_[us].Add(cap_base);
      if (update_auxiliary) hash_ ^= HashHand(us, hand_[us]);
    } else {
      undo.captured = Piece::None();
    }

    // Remove moving piece from source.
    Piece moved = piece_on(from);
    if (update_auxiliary) hash_ ^= Zobrist::Psq[moved.val][from.as_idx()];
    RemovePiece(from);

    // Promote if flagged.
    if (m.is_promotion()) {
      moved = moved.Promote();
    }

    // Place at destination.
    PutPiece(to, moved);
    if (update_auxiliary) hash_ ^= Zobrist::Psq[moved.val][to.as_idx()];
  }

  // Flip side to move.
  if (update_auxiliary) hash_ ^= Zobrist::Side;
  side_to_move_ = ~us;
  ply_++;

  // Update continuous check counter. Counted in plies (increment by 2
  // per check move), matching dlshogi's convention so that the counter
  // can be compared directly against the ply distance to the prior
  // occurrence in CheckRepetition().
  if (update_auxiliary) {
#ifndef NDEBUG
    if (gives_check >= 0) {
      assert((gives_check != 0) == InCheck(~us));
    }
#endif
    undo.gave_check = gives_check >= 0 ? gives_check != 0 : InCheck(~us);
    if (undo.gave_check) {
      continuous_check_[us] += 2;
    } else {
      continuous_check_[us] = 0;
    }
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
  if (undo.updated_auxiliary) {
    hash_ = undo.prev_hash;
    continuous_check_[us] = undo.prev_continuous_check;
  }

  // Remove the history entry we added in DoMove.
  if (undo.updated_auxiliary && !history_.empty()) {
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

void ShogiBoard::GenerateBoardMovesNonCheck(MoveList& moves,
                                            const Bitboard& pinned) const {
  Color us = side_to_move_;
  Square ksq = king_sq_[us];
  Bitboard our = pieces(us);
  Bitboard occ = occupied();
  Bitboard occ_without_king = occ ^ ShogiTables::SquareBB[ksq.as_idx()];

  Bitboard tmp = our;
  while (tmp.Any()) {
    Square from = tmp.Pop();
    Piece pc = piece_on(from);
    PieceType pt = pc.GetType();

    Bitboard targets = PieceAttacks(pt, us, from, occ) & ~our;

    if (from == ksq) {
      Bitboard legal_king_targets = Bitboard::Zero();
      Bitboard king_targets = targets;
      while (king_targets.Any()) {
        Square to = king_targets.Pop();
        if (!IsSquareAttacked(to, occ_without_king, ~us)) {
          legal_king_targets.Set(to);
        }
      }
      targets = legal_king_targets;
    } else if (pinned.Test(from)) {
      targets &= ShogiTables::LineBB[from.as_idx()][ksq.as_idx()];
    }

    targets.ForEach([&](Square to) {
      bool can_promote = false;
      bool must_promote = false;

      if (pt.CanPromote()) {
        can_promote = from.InPromotionZone(us) || to.InPromotionZone(us);

        Rank dest_rank = to.rank();
        Rank rel_rank = (us == BLACK) ? dest_rank
                                      : Rank::FromIdx(8 - dest_rank.idx);
        if (pt == kPawn || pt == kLance) {
          must_promote = (rel_rank.idx == 0);
        } else if (pt == kKnight) {
          must_promote = (rel_rank.idx <= 1);
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

Bitboard ShogiBoard::ComputeBlockersForKing(Color king_color) const {
  Square ksq = king_sq_[king_color];
  Bitboard occ = occupied();
  Bitboard blockers = Bitboard::Zero();
  Color enemy = ~king_color;

  // Rook-type pinners (rook, dragon) — check rank and file directions.
  Bitboard rook_pinners = pieces(enemy, kRook) | pieces(enemy, kDragon);
  Bitboard rook_rays = ShogiTables::RookEffectBB[ksq.as_idx()];
  Bitboard rook_candidates = rook_rays & rook_pinners;

  while (rook_candidates.Any()) {
    Square pinner = rook_candidates.Pop();
    Bitboard between = ShogiTables::BetweenBB[ksq.as_idx()][pinner.as_idx()] & occ;
    // Exactly one piece between king and pinner = that piece is pinned.
    if (between.Any() && !between.MoreThanOne()) {
      blockers |= between;
    }
  }

  // Bishop-type pinners (bishop, horse) — check diagonal directions.
  Bitboard bishop_pinners = pieces(enemy, kBishop) | pieces(enemy, kHorse);
  Bitboard bishop_rays = ShogiTables::BishopEffectBB[ksq.as_idx()];
  Bitboard bishop_candidates = bishop_rays & bishop_pinners;

  while (bishop_candidates.Any()) {
    Square pinner = bishop_candidates.Pop();
    Bitboard between = ShogiTables::BetweenBB[ksq.as_idx()][pinner.as_idx()] & occ;
    if (between.Any() && !between.MoreThanOne()) {
      blockers |= between;
    }
  }

  // Lance pinners — only pin along the file (vertical), color-dependent direction.
  Bitboard lance_pinners = pieces(enemy, kLance);
  // An enemy lance attacks toward the king from the opposite direction.
  // E.g., a BLACK lance moves up, so it must be BELOW the king to attack it.
  // Look from the king in the king's own forward direction to find enemy lances.
  Bitboard lance_rays = ShogiTables::LanceMaskBB[ksq.as_idx()][king_color];
  Bitboard lance_candidates = lance_rays & lance_pinners;

  while (lance_candidates.Any()) {
    Square pinner = lance_candidates.Pop();
    Bitboard between = ShogiTables::BetweenBB[ksq.as_idx()][pinner.as_idx()] & occ;
    if (between.Any() && !between.MoreThanOne()) {
      blockers |= between;
    }
  }

  return blockers;
}

MoveList ShogiBoard::GenerateLegalMoves() {
  Color us = side_to_move_;
  bool in_check = InCheck(us);

  if (!in_check) {
    Bitboard pinned = ComputeBlockersForKing(us) & pieces(us);

    MoveList legal;
    GenerateBoardMovesNonCheck(legal, pinned);

    MoveList drops;
    GenerateDropMoves(drops);
    for (const Move& m : drops) {
      if (m.drop_piece() == kPawn &&
          ShogiTables::PawnEffectBB[m.to().as_idx()][us].Test(king_sq_[~us])) {
        UndoInfo undo = DoMoveForMovegen(m);
        bool is_mate = !HasLegalEvasion();
        UndoMove(m, undo);
        if (is_mate) continue;
      }
      legal.push_back(m);
    }

    return legal;
  }

  return GenerateEvasionMoves();
}

MoveList ShogiBoard::GenerateEvasionMoves() {
  return GenerateEvasionMovesImpl(false);
}

bool ShogiBoard::HasLegalEvasion() {
  return !GenerateEvasionMovesImpl(true).empty();
}

MoveList ShogiBoard::GenerateEvasionMovesImpl(bool stop_after_one) {
  Color us = side_to_move_;
  Color them = ~us;
  Square ksq = king_sq_[us];
  Bitboard occ = occupied();
  Bitboard our = pieces(us);
  Bitboard checkers = AttackersTo(ksq, occ) & pieces(them);
  MoveList legal;

  // A double check can only be evaded by moving the king.
  Bitboard king_targets =
      ShogiTables::KingEffectBB[ksq.as_idx()] & ~our;
  Bitboard occ_without_king = occ;
  occ_without_king.Clear(ksq);
  while (king_targets.Any()) {
    Square to = king_targets.Pop();
    if (!IsSquareAttacked(to, occ_without_king, them)) {
      legal.push_back(Move::Normal(ksq, to));
      if (stop_after_one) return legal;
    }
  }

  if (checkers.MoreThanOne()) return legal;
  assert(checkers.Any());

  Square checker = checkers.Pop();
  Bitboard between =
      ShogiTables::BetweenBB[ksq.as_idx()][checker.as_idx()];
  Bitboard evasion_targets = between;
  evasion_targets.Set(checker);
  Bitboard pinned = ComputeBlockersForKing(us) & our;

  // A non-king evasion must capture the checker or interpose between
  // a sliding checker and the king.
  Bitboard candidates = our;
  candidates.Clear(ksq);
  while (candidates.Any()) {
    Square from = candidates.Pop();
    PieceType pt = piece_on(from).GetType();
    Bitboard targets =
        PieceAttacks(pt, us, from, occ) & ~our & evasion_targets;
    if (pinned.Test(from)) {
      targets &= ShogiTables::LineBB[from.as_idx()][ksq.as_idx()];
    }

    while (targets.Any()) {
      Square to = targets.Pop();
      bool can_promote =
          pt.CanPromote() &&
          (from.InPromotionZone(us) || to.InPromotionZone(us));
      bool must_promote = false;
      if (pt == kPawn || pt == kLance || pt == kKnight) {
        Rank dest_rank = to.rank();
        Rank rel_rank = us == BLACK ? dest_rank
                                    : Rank::FromIdx(8 - dest_rank.idx);
        must_promote = (pt == kKnight) ? rel_rank.idx <= 1
                                       : rel_rank.idx == 0;
      }

      if (can_promote) {
        legal.push_back(Move::Promotion(from, to));
        if (stop_after_one) return legal;
      }
      if (!must_promote) {
        legal.push_back(Move::Normal(from, to));
        if (stop_after_one) return legal;
      }
    }
  }

  // Drops can only block a single sliding check. Generate directly on
  // the usually small set of between-squares instead of enumerating
  // every possible drop on the board.
  Bitboard empty_between = between & ~occ;
  Bitboard our_pawns = pieces(us, kPawn);
  for (int pt_idx = kPawn.idx; pt_idx <= kGold.idx; ++pt_idx) {
    PieceType pt = PieceType::FromIdx(pt_idx);
    if (!hand_[us].Has(pt)) continue;

    Bitboard targets = empty_between;
    if (pt == kPawn || pt == kLance) {
      Rank forbidden = us == BLACK ? kRank1 : kRank9;
      targets &= ~ShogiTables::RankBB[forbidden.idx];
    }
    if (pt == kKnight) {
      if (us == BLACK) {
        targets &= ~ShogiTables::RankBB[0];
        targets &= ~ShogiTables::RankBB[1];
      } else {
        targets &= ~ShogiTables::RankBB[7];
        targets &= ~ShogiTables::RankBB[8];
      }
    }
    if (pt == kPawn) {
      for (int f = 0; f < 9; ++f) {
        if ((our_pawns & ShogiTables::FileBB[f]).Any()) {
          targets &= ~ShogiTables::FileBB[f];
        }
      }
    }

    while (targets.Any()) {
      Move move = Move::Drop(pt, targets.Pop());
      // A blocking pawn can exceptionally counter-check. Preserve the
      // pawn-drop-mate rule in that case.
      if (pt != kPawn || IsLegal(move, pinned)) {
        legal.push_back(move);
        if (stop_after_one) return legal;
      }
    }
  }

  return legal;
}

// =====================================================================
// GenerateCheckingMovesViaFilter: simple oracle implementation.
// =====================================================================
//
// Used as the reference implementation for the property test. Slow but
// obviously correct: for each legal move, classify "gives check" via
// bitboard ops + occasional do/undo for discovered-check candidates.
//
MoveList ShogiBoard::GenerateCheckingMovesViaFilter() {
  Color us = side_to_move_;
  Color them = ~us;
  Square king_sq = king_sq_[them];
  Bitboard occ = occupied();
  Bitboard our_blockers = ComputeBlockersForKing(them) & pieces(us);

  MoveList legal = GenerateLegalMoves();
  MoveList result;

  for (size_t i = 0; i < legal.size(); ++i) {
    const Move m = legal[i];
    bool gives_check = false;

    if (m.is_drop()) {
      PieceType pt = m.drop_piece();
      Square dst = m.to();
      Bitboard occ_after = occ;
      occ_after.Set(dst);
      gives_check = PieceAttacks(pt, us, dst, occ_after).Test(king_sq);
    } else {
      Square src = m.from();
      Square dst = m.to();
      PieceType pt = piece_on(src).GetType();
      if (m.is_promotion()) pt = pt.Promote();
      Bitboard occ_after = occ;
      occ_after.Clear(src);
      occ_after.Set(dst);
      if (PieceAttacks(pt, us, dst, occ_after).Test(king_sq)) {
        gives_check = true;
      } else if (our_blockers.Test(src)) {
        UndoInfo undo = DoMoveForMovegen(m);
        gives_check = InCheck();
        UndoMove(m, undo);
      }
    }
    if (gives_check) result.push_back(m);
  }
  return result;
}

// =====================================================================
// GenerateCheckingMoves: production fast version (Phase 7).
// =====================================================================
//
// Uses precomputed CheckBB tables to identify candidate pieces (pieces
// that COULD give direct or discovered check). For board moves, skips
// classification entirely on non-candidate pieces. For drops, looks up
// the per-piece "drop check zone" and only considers drops landing in
// that zone.
//
// The core saving over GenerateCheckingMovesViaFilter: most pieces in
// a typical position cannot possibly give check from any of their
// moves, so we avoid the per-move PieceAttacks classification call.
//
MoveList ShogiBoard::GenerateCheckingMoves() {
  Color us = side_to_move_;

  if (InCheck(us)) {
    return GenerateCheckingMovesViaFilter();
  }
  return GenerateCheckingMovesNonCheck();
}

MoveList ShogiBoard::GenerateCheckingMovesNonCheck() {
  Color us = side_to_move_;
  Color them = ~us;
  Square king_sq = king_sq_[them];
  int ksq_idx = king_sq.as_idx();
  Bitboard occ = occupied();
  Bitboard our = pieces(us);
  Bitboard pinned = Bitboard::Zero();
  bool pinned_computed = false;

  // Pieces that, if moved, may discover check.
  Bitboard our_blockers = ComputeBlockersForKing(them) & our;

  // Direct-check candidates: pieces whose source square is in the
  // promotion-aware MoveCheckBB. A piece NOT in this set can't possibly
  // give direct check via any move (incl. promotion). Rooks/dragons
  // are universal candidates (no zone constraint).
  Bitboard direct_candidates =
      (pieces(us, kPawn)   & ShogiTables::PawnMoveCheckBB[ksq_idx][us])   |
      (pieces(us, kLance)  & ShogiTables::LanceMoveCheckBB[ksq_idx][us])  |
      (pieces(us, kKnight) & ShogiTables::KnightMoveCheckBB[ksq_idx][us]) |
      (pieces(us, kSilver) & ShogiTables::SilverMoveCheckBB[ksq_idx][us]) |
      ((pieces(us, kGold)     | pieces(us, kProPawn) |
        pieces(us, kProLance) | pieces(us, kProKnight) |
        pieces(us, kProSilver)) & ShogiTables::GoldMoveCheckBB[ksq_idx][us]) |
      (pieces(us, kBishop) & ShogiTables::BishopMoveCheckBB[ksq_idx][us]) |
      (pieces(us, kHorse)  & ShogiTables::HorseMoveCheckBB[ksq_idx])      |
      pieces(us, kRook) | pieces(us, kDragon);

  MoveList result;

  auto direct_check_zone = [&](PieceType pt) {
    switch (pt.idx) {
      case kPawn.idx:
        return ShogiTables::PawnCheckBB[ksq_idx][us];
      case kLance.idx:
        return ShogiTables::LanceCheckBB[ksq_idx][us];
      case kKnight.idx:
        return ShogiTables::KnightCheckBB[ksq_idx][us];
      case kSilver.idx:
        return ShogiTables::SilverCheckBB[ksq_idx][us];
      case kGold.idx:
      case kProPawn.idx:
      case kProLance.idx:
      case kProKnight.idx:
      case kProSilver.idx:
        return ShogiTables::GoldCheckBB[ksq_idx][us];
      case kBishop.idx:
        return ShogiTables::BishopCheckBB[ksq_idx];
      case kRook.idx:
        return ShogiTables::RookEffectBB[ksq_idx];
      case kKing.idx:
        return ShogiTables::KingEffectBB[ksq_idx];
      case kHorse.idx:
        return ShogiTables::HorseCheckBB[ksq_idx];
      case kDragon.idx:
        return ShogiTables::RookEffectBB[ksq_idx] |
               ShogiTables::DragonStepBB[ksq_idx];
      default:
        return Bitboard::Zero();
    }
  };

  auto possible_direct_to = [&](PieceType pt, Square from) {
    Bitboard zone = direct_check_zone(pt);
    if (pt.CanPromote()) {
      Bitboard promoted_zone = direct_check_zone(pt.Promote());
      if (!from.InPromotionZone(us)) {
        promoted_zone &= ShogiTables::PromotionZoneBB[us];
      }
      zone |= promoted_zone;
    }
    return zone;
  };

  auto directly_attacks = [&](PieceType pt, Square to,
                              const Bitboard& occ_after) {
    if (!direct_check_zone(pt).Test(to)) return false;
    switch (pt.idx) {
      case kPawn.idx:
      case kKnight.idx:
      case kSilver.idx:
      case kGold.idx:
      case kProPawn.idx:
      case kProLance.idx:
      case kProKnight.idx:
      case kProSilver.idx:
      case kKing.idx:
        return true;
      default:
        return PieceAttacks(pt, us, to, occ_after).Test(king_sq);
    }
  };

  auto must_promote = [&](PieceType pt, Square to) {
    if (pt != kPawn && pt != kLance && pt != kKnight) return false;
    Rank dest_rank = to.rank();
    Rank rel_rank = us == BLACK ? dest_rank
                                : Rank::FromIdx(8 - dest_rank.idx);
    if (pt == kPawn || pt == kLance) return rel_rank.idx == 0;
    return rel_rank.idx <= 1;
  };

  auto can_promote = [&](PieceType pt, Square from, Square to) {
    return pt.CanPromote() &&
           (from.InPromotionZone(us) || to.InPromotionZone(us));
  };

  auto add_if_legal = [&](Move move) {
    if (!pinned_computed) {
      pinned = ComputeBlockersForKing(us) & our;
      pinned_computed = true;
    }
    if (IsLegal(move, pinned)) result.push_back(move);
  };

  auto add_variants = [&](Square from, Square to, PieceType pt,
                          const auto& gives_check) {
    if (can_promote(pt, from, to)) {
      PieceType promoted = pt.Promote();
      if (gives_check(promoted)) add_if_legal(Move::Promotion(from, to));
    }
    if (!must_promote(pt, to) && gives_check(pt)) {
      add_if_legal(Move::Normal(from, to));
    }
  };

  auto add_direct_variants = [&](Square from, Square to, PieceType pt) {
    Bitboard occ_after = occ;
    occ_after.Clear(from);
    occ_after.Set(to);
    add_variants(from, to, pt, [&](PieceType moved_pt) {
      return directly_attacks(moved_pt, to, occ_after);
    });
  };

  auto add_discovered_variants = [&](Square from, Square to, PieceType pt) {
    add_variants(from, to, pt, [](PieceType) { return true; });
  };

  Bitboard discovered = our_blockers;
  while (discovered.Any()) {
    Square from = discovered.Pop();
    PieceType pt = piece_on(from).GetType();
    Bitboard targets = PieceAttacks(pt, us, from, occ) & ~our;
    const Bitboard line = ShogiTables::LineBB[king_sq.as_idx()][from.as_idx()];

    Bitboard discovered_targets = targets & ~line;
    while (discovered_targets.Any()) {
      add_discovered_variants(from, discovered_targets.Pop(), pt);
    }

    if (direct_candidates.Test(from)) {
      Bitboard direct_targets = targets & line & possible_direct_to(pt, from);
      while (direct_targets.Any()) {
        add_direct_variants(from, direct_targets.Pop(), pt);
      }
    }
  }

  Bitboard direct = direct_candidates & ~our_blockers;
  while (direct.Any()) {
    Square from = direct.Pop();
    PieceType pt = piece_on(from).GetType();
    Bitboard targets =
        PieceAttacks(pt, us, from, occ) & ~our & possible_direct_to(pt, from);
    while (targets.Any()) {
      add_direct_variants(from, targets.Pop(), pt);
    }
  }

  auto drop_targets = [&](PieceType pt) {
    Bitboard targets = ~occ & Bitboard::All();

    if (pt == kPawn || pt == kLance) {
      targets &= ~(us == BLACK ? ShogiTables::RankBB[0]
                               : ShogiTables::RankBB[8]);
    }
    if (pt == kKnight) {
      if (us == BLACK) {
        targets &= ~ShogiTables::RankBB[0];
        targets &= ~ShogiTables::RankBB[1];
      } else {
        targets &= ~ShogiTables::RankBB[7];
        targets &= ~ShogiTables::RankBB[8];
      }
    }
    if (pt == kPawn) {
      Bitboard pawns = pieces(us, kPawn);
      for (int f = 0; f < 9; ++f) {
        if ((pawns & ShogiTables::FileBB[f]).Any()) {
          targets &= ~ShogiTables::FileBB[f];
        }
      }
    }

    return targets;
  };

  auto add_drop_checks = [&](PieceType pt, Bitboard check_zone,
                             bool needs_occlusion_check) {
    if (!hand_[us].Has(pt)) return;
    Bitboard targets = drop_targets(pt) & check_zone;
    while (targets.Any()) {
      Square to = targets.Pop();
      if (needs_occlusion_check) {
        Bitboard occ_after = occ;
        occ_after.Set(to);
        if (!PieceAttacks(pt, us, to, occ_after).Test(king_sq)) continue;
      }
      Move move = Move::Drop(pt, to);
      if (IsLegal(move, pinned)) result.push_back(move);
    }
  };

  add_drop_checks(kPawn, ShogiTables::PawnCheckBB[ksq_idx][us], false);
  add_drop_checks(kLance, ShogiTables::LanceCheckBB[ksq_idx][us], true);
  add_drop_checks(kKnight, ShogiTables::KnightCheckBB[ksq_idx][us], false);
  add_drop_checks(kSilver, ShogiTables::SilverCheckBB[ksq_idx][us], false);
  add_drop_checks(kGold, ShogiTables::GoldCheckBB[ksq_idx][us], false);
  add_drop_checks(kBishop, ShogiTables::BishopCheckBB[ksq_idx], true);
  add_drop_checks(kRook, ShogiTables::RookEffectBB[ksq_idx], true);

  return result;
}

// =====================================================================
// Hand-specialized mate-in-1.
//
// Adapted from dlshogi/YaneuraOu's Position::mateMoveIn1Ply at
// DeepLearningShogi commit 5bdf2c8c7ae664651204f29fdbc3d1f2937a8135.
// The candidate walk is fused with analytical evasion tests: it never
// creates a checking-move list and never makes a full board move.
// =====================================================================

Move ShogiBoard::FindMateInOne() {
  if (!king_sq_[BLACK].IsValid() || !king_sq_[WHITE].IsValid()) {
    return Move();
  }
  if (!InCheck()) return FindMateInOneNonCheck();

  // The upstream specialized routine requires a non-check position.
  // Preserve the complete legal checking-move path for counterchecks.
  MoveList checks = GenerateCheckingMoves();
  for (const Move& move : checks) {
    UndoInfo undo = DoMove(move, true);
    const bool mate = !HasLegalEvasion();
    UndoMove(move, undo);
    if (mate) return move;
  }
  return Move();
}

Move ShogiBoard::FindMateInOneNonCheck() {
  if (!king_sq_[BLACK].IsValid() || !king_sq_[WHITE].IsValid()) {
    return Move();
  }
  assert(!InCheck());
  Move result = side_to_move_ == BLACK
                    ? FindMateInOneNonCheckImpl<BLACK>()
                    : FindMateInOneNonCheckImpl<WHITE>();

#ifndef NDEBUG
  if (!result.is_null()) {
    MoveList legal = GenerateLegalMoves();
    bool found = false;
    for (const Move& move : legal) {
      if (move == result) {
        found = true;
        break;
      }
    }
    assert(found);
    UndoInfo undo = DoMove(result, true);
    assert(!HasLegalEvasion());
    UndoMove(result, undo);
  }
#endif

  return result;
}

bool ShogiBoard::CanKingEscapeAfterMateProbe(
    Color attacker, Square checker_square, const Bitboard& occupied_after,
    const Bitboard& moved_checker_attacks) const {
  const Color defender = ~attacker;
  const Square king_square = king_sq_[defender];

  Bitboard targets =
      ShogiTables::KingEffectBB[king_square.as_idx()] & ~pieces(defender);

  // The moved checker is deliberately absent from the persistent piece
  // bitboards. Add its square explicitly as a possible king capture.
  if (ShogiTables::KingEffectBB[king_square.as_idx()].Test(checker_square)) {
    targets.Set(checker_square);
  }

  Bitboard occupied_without_king = occupied_after;
  occupied_without_king.Clear(king_square);

  while (targets.Any()) {
    const Square to = targets.Pop();
    Bitboard escape_occupied = occupied_without_king;

    if (to == checker_square) {
      // The moved checker is not installed in the persistent piece
      // bitboards, so clearing its square gives the exact post-capture
      // occupancy for attacks by all supporting pieces.
      escape_occupied.Clear(checker_square);
      if (!IsSquareAttacked(to, escape_occupied, attacker)) return true;
      continue;
    }

    if (moved_checker_attacks.Test(to)) continue;
    if (!IsSquareAttacked(to, escape_occupied, attacker)) return true;
  }

  return false;
}

bool ShogiBoard::CanDefenderCaptureMateChecker(
    Color defender, Square checker_square,
    const Bitboard& occupied_after) const {
  Bitboard candidates =
      AttackersTo(checker_square, occupied_after) & pieces(defender);
  candidates.Clear(king_sq_[defender]);
  if (candidates.Empty()) return false;

  const Square king_square = king_sq_[defender];
  const Bitboard pinned =
      ComputeBlockersForKing(defender) & pieces(defender);
  while (candidates.Any()) {
    const Square from = candidates.Pop();
    if (!pinned.Test(from) ||
        ShogiTables::LineBB[from.as_idx()][king_square.as_idx()]
            .Test(checker_square)) {
      return true;
    }
  }
  return false;
}

bool ShogiBoard::CanDefenderInterposeMateCheck(
    Color defender, const Bitboard& between,
    const Bitboard& occupied_after) const {
  if (between.Empty()) return false;

  const Bitboard defenders = pieces(defender);
  const Square king_square = king_sq_[defender];
  const Bitboard pinned =
      ComputeBlockersForKing(defender) & defenders;

  Bitboard sources = defenders;
  sources.Clear(king_square);
  while (sources.Any()) {
    const Square from = sources.Pop();
    const PieceType type = piece_on(from).GetType();
    Bitboard targets =
        PieceAttacks(type, defender, from, occupied_after) & between;
    if (targets.Empty()) continue;

    if (!pinned.Test(from)) return true;
    targets &=
        ShogiTables::LineBB[from.as_idx()][king_square.as_idx()];
    if (targets.Any()) return true;
  }

  Bitboard empty_between = between & ~occupied_after;
  if (empty_between.Empty()) return false;

  const Hand defender_hand = hand_[defender];
  const Bitboard defender_pawns = pieces(defender, kPawn);
  for (int type_idx = kPawn.idx; type_idx <= kGold.idx; ++type_idx) {
    const PieceType type = PieceType::FromIdx(type_idx);
    if (!defender_hand.Has(type)) continue;

    Bitboard targets = empty_between;
    if (type == kPawn || type == kLance) {
      targets &= ~(defender == BLACK ? ShogiTables::RankBB[0]
                                     : ShogiTables::RankBB[8]);
    }
    if (type == kKnight) {
      if (defender == BLACK) {
        targets &= ~ShogiTables::RankBB[0];
        targets &= ~ShogiTables::RankBB[1];
      } else {
        targets &= ~ShogiTables::RankBB[7];
        targets &= ~ShogiTables::RankBB[8];
      }
    }
    if (type == kPawn) {
      for (int file = 0; file < 9; ++file) {
        if ((defender_pawns & ShogiTables::FileBB[file]).Any()) {
          targets &= ~ShogiTables::FileBB[file];
        }
      }
    }
    if (targets.Any()) return true;
  }

  return false;
}

template <Color Us>
bool ShogiBoard::IsMateAfterMateProbe(PieceType moved_type,
                                      Square checker_square) {
  constexpr Color Them = Us == BLACK ? WHITE : BLACK;
  const Square king_square = king_sq_[Them];
  MateProbeCaptureRemoval capture_removal(*this, checker_square);

  Bitboard occupied_after = occupied();
  occupied_after.Set(checker_square);
  Bitboard occupied_without_king = occupied_after;
  occupied_without_king.Clear(king_square);

  const Bitboard moved_checker_attacks =
      PieceAttacks(moved_type, Us, checker_square, occupied_without_king);
  const bool direct_check = moved_checker_attacks.Test(king_square);

  // The moved checker is deliberately absent from the persistent
  // bitboards. Any attacker found here is a discovered checker.
  Bitboard discovered_checkers =
      AttackersTo(king_square, occupied_after) & pieces(Us);
  if (!direct_check && discovered_checkers.Empty()) return false;

  if (CanKingEscapeAfterMateProbe(
          Us, checker_square, occupied_after, moved_checker_attacks)) {
    return false;
  }

  const int checker_count =
      discovered_checkers.PopCount() + (direct_check ? 1 : 0);
  if (checker_count > 1) {
    // As with ordinary double check, only a king move can resolve both
    // the moved checker and the discovered checker.
    return true;
  }

  const Square sole_checker =
      direct_check ? checker_square : discovered_checkers.Peek();
  if (CanDefenderCaptureMateChecker(Them, sole_checker, occupied_after)) {
    return false;
  }

  const Bitboard between =
      ShogiTables::BetweenBB[king_square.as_idx()][sole_checker.as_idx()];
  if (CanDefenderInterposeMateCheck(Them, between, occupied_after)) {
    return false;
  }

  return true;
}

template <Color Us>
Move ShogiBoard::FindMateInOneNonCheckImpl() {
  constexpr Color Them = Us == BLACK ? WHITE : BLACK;
  const Square king_square = king_sq_[Them];
  const int king_idx = king_square.as_idx();
  const Bitboard occupied = this->occupied();
  const Bitboard our_pieces = pieces(Us);

  auto try_drops = [&](PieceType type, Bitboard check_zone) {
    if (!hand_[Us].Has(type)) return Move();
    Bitboard targets = check_zone & ~occupied;
    while (targets.Any()) {
      const Square to = targets.Pop();
      if (IsMateAfterMateProbe<Us>(type, to)) {
        return Move::Drop(type, to);
      }
    }
    return Move();
  };

  // Preserve the upstream ordering: major drops first, followed by
  // gold/silver/knight. Pawn-drop mate is illegal and is never tried.
  Move mate = try_drops(kRook, ShogiTables::RookEffectBB[king_idx]);
  if (!mate.is_null()) return mate;
  mate = try_drops(kLance, ShogiTables::LanceCheckBB[king_idx][Us]);
  if (!mate.is_null()) return mate;
  mate = try_drops(kBishop, ShogiTables::BishopCheckBB[king_idx]);
  if (!mate.is_null()) return mate;
  mate = try_drops(kGold, ShogiTables::GoldCheckBB[king_idx][Us]);
  if (!mate.is_null()) return mate;
  mate = try_drops(kSilver, ShogiTables::SilverCheckBB[king_idx][Us]);
  if (!mate.is_null()) return mate;
  mate = try_drops(kKnight, ShogiTables::KnightCheckBB[king_idx][Us]);
  if (!mate.is_null()) return mate;

  const Bitboard discovered =
      ComputeBlockersForKing(Them) & our_pieces;
  const Bitboard pinned =
      ComputeBlockersForKing(Us) & our_pieces;

  Bitboard direct_candidates =
      (pieces(Us, kPawn) &
       ShogiTables::PawnMoveCheckBB[king_idx][Us]) |
      (pieces(Us, kLance) &
       ShogiTables::LanceMoveCheckBB[king_idx][Us]) |
      (pieces(Us, kKnight) &
       ShogiTables::KnightMoveCheckBB[king_idx][Us]) |
      (pieces(Us, kSilver) &
       ShogiTables::SilverMoveCheckBB[king_idx][Us]) |
      ((pieces(Us, kGold) | pieces(Us, kProPawn) |
        pieces(Us, kProLance) | pieces(Us, kProKnight) |
        pieces(Us, kProSilver)) &
       ShogiTables::GoldMoveCheckBB[king_idx][Us]) |
      (pieces(Us, kBishop) &
       ShogiTables::BishopMoveCheckBB[king_idx][Us]) |
      (pieces(Us, kHorse) &
       ShogiTables::HorseMoveCheckBB[king_idx]) |
      pieces(Us, kRook) | pieces(Us, kDragon);

  const Bitboard candidates = direct_candidates | discovered;
  static constexpr int kPieceOrder[] = {
      kDragon.idx, kRook.idx, kHorse.idx, kBishop.idx,
      kGold.idx, kProPawn.idx, kProLance.idx, kProKnight.idx,
      kProSilver.idx, kSilver.idx, kKnight.idx, kLance.idx,
      kPawn.idx, kKing.idx,
  };

  auto must_promote = [&](PieceType type, Square to) {
    if (type != kPawn && type != kLance && type != kKnight) return false;
    const Rank rank = to.rank();
    const Rank relative =
        Us == BLACK ? rank : Rank::FromIdx(8 - rank.idx);
    return type == kKnight ? relative.idx <= 1 : relative.idx == 0;
  };

  auto can_promote = [&](PieceType type, Square from, Square to) {
    return type.CanPromote() &&
           (from.InPromotionZone(Us) || to.InPromotionZone(Us));
  };

  for (const int type_idx : kPieceOrder) {
    const PieceType type = PieceType::FromIdx(type_idx);
    Bitboard sources = pieces(Us, type) & candidates;
    while (sources.Any()) {
      const Square from = sources.Pop();
      Bitboard targets =
          PieceAttacks(type, Us, from, occupied) & ~our_pieces;
      const bool source_discovers = discovered.Test(from);
      const Bitboard check_line =
          ShogiTables::LineBB[from.as_idx()][king_idx];

      if (pinned.Test(from)) {
        targets &=
            ShogiTables::LineBB[from.as_idx()][king_sq_[Us].as_idx()];
      }
      if (type == kKing) {
        Bitboard legal_king_targets = Bitboard::Zero();
        while (targets.Any()) {
          const Square to = targets.Pop();
          if (IsLegal(Move::Normal(from, to), pinned)) {
            legal_king_targets.Set(to);
          }
        }
        targets = legal_king_targets;
      }

      MateProbeSourceRemoval source_removal(*this, Us, type, from);
      while (targets.Any()) {
        const Square to = targets.Pop();
        const bool discovered_check =
            source_discovers && !check_line.Test(to);
        Bitboard occupied_after = this->occupied();
        occupied_after.Set(to);
        Bitboard occupied_without_king = occupied_after;
        occupied_without_king.Clear(king_square);

        auto try_variant = [&](PieceType moved_type, Move move) {
          const bool direct_check =
              PieceAttacks(moved_type, Us, to, occupied_without_king)
                  .Test(king_square);
          if (!direct_check && !discovered_check) return Move();
          return IsMateAfterMateProbe<Us>(moved_type, to) ? move : Move();
        };

        if (can_promote(type, from, to)) {
          mate = try_variant(
              type.Promote(), Move::Promotion(from, to));
          if (!mate.is_null()) return mate;
        }
        if (!must_promote(type, to)) {
          mate = try_variant(type, Move::Normal(from, to));
          if (!mate.is_null()) return mate;
        }
      }
    }
  }

  return Move();
}

bool ShogiBoard::IsLegal(Move m, const Bitboard& pinned) {
  Color us = side_to_move_;
  Square ksq = king_sq_[us];

  if (m.is_drop()) {
    // Drop moves are always legal (nifu, rank restrictions already handled
    // in GenerateDropMoves). Exception: pawn drop checkmate (uchifuzume).
    if (m.drop_piece() == kPawn) {
      // Check if this pawn drop gives check.
      Square to = m.to();
      if (ShogiTables::PawnEffectBB[to.as_idx()][us].Test(king_sq_[~us])) {
        // The pawn drop gives check. Use DoMove to check if it's checkmate.
        UndoInfo undo = DoMoveForMovegen(m);
        bool is_mate = !HasLegalEvasion();
        UndoMove(m, undo);
        if (is_mate) return false;  // Pawn drop checkmate is illegal.
      }
    }
    return true;
  }

  Square from = m.from();
  Square to = m.to();

  // King moves: check if destination is attacked by enemy.
  if (from == ksq) {
    // Remove king from occupancy to detect X-ray attacks through king's source.
    Bitboard occ_without_king = occupied() ^ ShogiTables::SquareBB[ksq.as_idx()];
    return (AttackersTo(to, occ_without_king) & pieces(~us)).Empty();
  }

  // Non-king moves: legal unless the piece is pinned and moves off the pin line.
  if (pinned.Empty()) return true;  // No pinned pieces at all — fast path.
  if (!pinned.Test(from)) return true;  // This piece is not pinned.

  // Piece is pinned: legal only if it moves along the pin line (from-king line).
  return ShogiTables::LineBB[from.as_idx()][ksq.as_idx()].Test(to);
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
  // Detect repetition at the FIRST prior occurrence (search-time policy
  // matching dlshogi and YaneuraOu). The actual rule fires on the 4th
  // occurrence, but treating the 2nd occurrence as already-perpetual
  // lets the search avoid the cycle with shallow lookahead — otherwise
  // the engine has to look 12+ plies ahead to see the 4-fold and may
  // walk into OUTE_SENNICHITE losses (see oute_sennichite1.csa).
  //
  // dist = ply distance to the prior occurrence. continuous_check_[c]
  // is also in plies (incremented by 2 per check). If our side has
  // been continuously checking for the entire cycle, we LOSE; if the
  // opponent has, we WIN; otherwise it's a normal draw.

  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
    const auto& entry = history_[i];
    if (entry.hash == hash_ &&
        entry.hand_black == hand_[BLACK].raw() &&
        entry.hand_white == hand_[WHITE].raw()) {
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
