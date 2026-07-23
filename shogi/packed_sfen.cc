/*
  YaneuraOu-compatible 32-byte PackedSfen codec.

  The bit stream is written least-significant bit first:
    side-to-move (1), black/white king squares (7 each), board pieces,
    hand pieces, then missing physical pieces ("piece box") to 256 bits.
*/

#include "shogi/board.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace lczero {
namespace {

struct HuffmanCode {
  uint8_t code;
  uint8_t bits;
};

constexpr std::array<HuffmanCode, 8> kBoardCodes{{
    {0x00, 1},  // empty
    {0x01, 2},  // pawn
    {0x03, 4},  // lance
    {0x0b, 4},  // knight
    {0x07, 4},  // silver
    {0x1f, 6},  // bishop
    {0x3f, 6},  // rook
    {0x0f, 5},  // gold
}};

constexpr std::array<HuffmanCode, 8> kPieceBoxCodes{{
    {0x00, 1},
    {0x02, 2},  // pawn
    {0x09, 4},  // lance
    {0x0d, 4},  // knight
    {0x0b, 4},  // silver
    {0x2f, 6},  // bishop
    {0x3f, 6},  // rook
    {0x1b, 5},  // gold
}};

// Physical piece totals in a complete shogi set, indexed by raw piece type.
constexpr std::array<int, 8> kPhysicalPieceCounts{
    0, 18, 4, 4, 4, 2, 2, 4,
};

// cshogi/YaneuraOu hand serialization order.
constexpr std::array<PieceType, 7> kHandOrder{
    kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook,
};

class BitWriter {
 public:
  explicit BitWriter(PackedSfen* packed) : packed_(packed) {
    packed_->data.fill(0);
  }

  bool Write(uint32_t value, int bits) {
    if (bits < 0 || cursor_ + bits > 256) return false;
    for (int i = 0; i < bits; ++i) {
      if ((value >> i) & 1U) {
        packed_->data[static_cast<size_t>(cursor_ >> 3)] |=
            static_cast<uint8_t>(1U << (cursor_ & 7));
      }
      ++cursor_;
    }
    return true;
  }

  int cursor() const { return cursor_; }

 private:
  PackedSfen* packed_;
  int cursor_ = 0;
};

class BitReader {
 public:
  explicit BitReader(const PackedSfen& packed) : packed_(packed) {}

  bool Read(int bits, uint32_t* value) {
    if (value == nullptr || bits < 0 || cursor_ + bits > 256) return false;
    *value = 0;
    for (int i = 0; i < bits; ++i) {
      const uint8_t byte = packed_.data[static_cast<size_t>(cursor_ >> 3)];
      *value |= uint32_t((byte >> (cursor_ & 7)) & 1U) << i;
      ++cursor_;
    }
    return true;
  }

  bool ReadBit(bool* value) {
    uint32_t bit = 0;
    if (!Read(1, &bit)) return false;
    *value = bit != 0;
    return true;
  }

  int cursor() const { return cursor_; }

 private:
  const PackedSfen& packed_;
  int cursor_ = 0;
};

bool WriteBoardPiece(BitWriter* writer, Piece piece) {
  if (piece.IsNone()) {
    return writer->Write(kBoardCodes[0].code, kBoardCodes[0].bits);
  }

  PieceType type = piece.GetType();
  const bool promoted = type.IsPromoted();
  const PieceType raw = promoted ? type.Unpromote() : type;
  if (raw.idx < 1 || raw.idx > 7) return false;

  const auto code = kBoardCodes[raw.idx];
  if (!writer->Write(code.code, code.bits)) return false;
  if (raw != kGold && !writer->Write(promoted ? 1 : 0, 1)) return false;
  return writer->Write(piece.GetColor() == WHITE ? 1 : 0, 1);
}

bool WriteHandPiece(BitWriter* writer, Color color, PieceType raw) {
  if (!raw.IsHandPiece()) return false;
  const auto code = kBoardCodes[raw.idx];
  if (!writer->Write(code.code >> 1, code.bits - 1)) return false;
  if (raw != kGold && !writer->Write(0, 1)) return false;
  return writer->Write(color == WHITE ? 1 : 0, 1);
}

bool WritePieceBoxPiece(BitWriter* writer, PieceType raw) {
  if (!raw.IsHandPiece()) return false;
  const auto code = kPieceBoxCodes[raw.idx];
  if (!writer->Write(code.code, code.bits)) return false;
  // Non-gold piece-box codes still need the trailing color bit. Gold's code
  // already decodes as a promoted white hand piece and consumes all 5 bits.
  return raw == kGold || writer->Write(0, 1);
}

bool ReadBoardType(BitReader* reader, PieceType* type) {
  uint32_t code = 0;
  for (int bits = 1; bits <= 6; ++bits) {
    uint32_t bit = 0;
    if (!reader->Read(1, &bit)) return false;
    code |= bit << (bits - 1);
    for (uint8_t raw = 0; raw < kBoardCodes.size(); ++raw) {
      if (kBoardCodes[raw].bits == bits && kBoardCodes[raw].code == code) {
        *type = PieceType::FromIdx(raw);
        return true;
      }
    }
  }
  return false;
}

bool ReadBoardPiece(BitReader* reader, Piece* piece) {
  PieceType raw = kNoPieceType;
  if (!ReadBoardType(reader, &raw)) return false;
  if (raw == kNoPieceType) {
    *piece = Piece::None();
    return true;
  }

  bool promoted = false;
  if (raw != kGold && !reader->ReadBit(&promoted)) return false;
  bool white = false;
  if (!reader->ReadBit(&white)) return false;
  PieceType type = promoted ? raw.Promote() : raw;
  *piece = Piece::Make(white ? WHITE : BLACK, type);
  return true;
}

bool ReadHandPiece(BitReader* reader, PieceType* raw, Color* color,
                   bool* piece_box) {
  uint32_t code = 0;
  PieceType decoded = kNoPieceType;
  for (int bits = 1; bits <= 5; ++bits) {
    uint32_t bit = 0;
    if (!reader->Read(1, &bit)) return false;
    code |= bit << (bits - 1);
    for (uint8_t candidate = 1; candidate <= 7; ++candidate) {
      const auto huffman = kBoardCodes[candidate];
      if (huffman.bits - 1 == bits &&
          (huffman.code >> 1) == code) {
        decoded = PieceType::FromIdx(candidate);
        break;
      }
    }
    if (decoded != kNoPieceType) break;
  }
  if (decoded == kNoPieceType) return false;

  bool promoted = false;
  if (decoded != kGold && !reader->ReadBit(&promoted)) return false;
  bool white = false;
  if (!reader->ReadBit(&white)) return false;

  *raw = decoded;
  *color = white ? WHITE : BLACK;
  *piece_box = promoted;
  return true;
}

}  // namespace

bool ShogiBoard::ToPackedSfen(PackedSfen* packed) const {
  if (packed == nullptr) return false;

  BitWriter writer(packed);
  if (!writer.Write(side_to_move_ == WHITE ? 1 : 0, 1)) return false;
  if (!writer.Write(king_sq_[BLACK].as_idx(), 7) ||
      !writer.Write(king_sq_[WHITE].as_idx(), 7)) {
    return false;
  }

  auto remaining = kPhysicalPieceCounts;
  for (int square = 0; square < kSquareNB; ++square) {
    const Piece piece = board_[square];
    if (!piece.IsNone() && piece.GetType() == kKing) continue;
    if (!WriteBoardPiece(&writer, piece)) return false;
    if (!piece.IsNone()) {
      const PieceType raw = piece.GetType().IsPromoted()
                                ? piece.GetType().Unpromote()
                                : piece.GetType();
      if (raw.idx < 1 || raw.idx > 7 || --remaining[raw.idx] < 0) {
        return false;
      }
    }
  }

  for (Color color : {BLACK, WHITE}) {
    for (PieceType raw : kHandOrder) {
      const int count = hand_[color].Count(raw);
      remaining[raw.idx] -= count;
      if (remaining[raw.idx] < 0) return false;
      for (int i = 0; i < count; ++i) {
        if (!WriteHandPiece(&writer, color, raw)) return false;
      }
    }
  }

  for (PieceType raw : kHandOrder) {
    for (int i = 0; i < remaining[raw.idx]; ++i) {
      if (!WritePieceBoxPiece(&writer, raw)) return false;
    }
  }

  return writer.cursor() == 256;
}

bool ShogiBoard::SetFromPackedSfen(const PackedSfen& packed, int game_ply) {
  BitReader reader(packed);
  uint32_t turn = 0;
  uint32_t black_king = 0;
  uint32_t white_king = 0;
  if (!reader.Read(1, &turn) || !reader.Read(7, &black_king) ||
      !reader.Read(7, &white_king)) {
    return false;
  }
  if (turn > 1 || black_king > 81 || white_king > 81 ||
      (black_king < 81 && black_king == white_king)) {
    return false;
  }

  *this = ShogiBoard();
  side_to_move_ = turn == 0 ? BLACK : WHITE;
  ply_ = std::max(1, game_ply);

  if (black_king < 81) {
    PutPiece(Square::FromIdx(static_cast<uint8_t>(black_king)),
             Piece::Make(BLACK, kKing));
  }
  if (white_king < 81) {
    PutPiece(Square::FromIdx(static_cast<uint8_t>(white_king)),
             Piece::Make(WHITE, kKing));
  }

  for (int square = 0; square < kSquareNB; ++square) {
    if (square == static_cast<int>(black_king) ||
        square == static_cast<int>(white_king)) {
      continue;
    }
    Piece piece = Piece::None();
    if (!ReadBoardPiece(&reader, &piece)) return false;
    if (!piece.IsNone()) {
      PutPiece(Square::FromIdx(static_cast<uint8_t>(square)), piece);
    }
  }

  while (reader.cursor() < 256) {
    PieceType raw = kNoPieceType;
    Color color = BLACK;
    bool piece_box = false;
    if (!ReadHandPiece(&reader, &raw, &color, &piece_box)) return false;
    if (!piece_box) hand_[color].Add(raw);
  }
  if (reader.cursor() != 256) return false;

  ComputeHash();
  ClearHistory();
  return true;
}

}  // namespace lczero
