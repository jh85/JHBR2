#include <atomic>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "dlshogi_mcts/uct_node.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace {

using dlshogi_mcts::NodeTree;
using dlshogi_mcts::uct_node_t;
using lczero::Move;
using lczero::ShogiBoard;

int failures = 0;

void Check(const char* name, bool condition) {
  if (!condition) {
    std::printf("  FAIL  %s\n", name);
    ++failures;
  }
}

uct_node_t* AppendChild(uct_node_t* parent, Move move, int visits) {
  parent->CreateSingleChildNode(move);
  auto* child = parent->CreateChildNode(0);
  child->move_count.store(visits, std::memory_order_relaxed);
  return child;
}

void TestExtensionAndIdenticalPosition() {
  constexpr uint64_t kStartKey = 0x1234;
  const Move move1 = Move::Parse("7g7f");
  const Move move2 = Move::Parse("3c3d");
  const Move move3 = Move::Parse("2g2f");

  NodeTree tree;
  Check("first position is not reused",
        !tree.ResetToPosition(kStartKey, {}));

  auto* start = tree.GetCurrentHead();
  auto* node1 = AppendChild(start, move1, 11);
  auto* start_children = start->child.get();
  auto* start_child_nodes = start->child_nodes.get();
  auto* node2 = AppendChild(node1, move2, 37);
  auto* node1_children = node1->child.get();
  auto* node1_child_nodes = node1->child_nodes.get();
  node2->win.store(23.0f, std::memory_order_relaxed);

  Check("extended position reuses tree",
        tree.ResetToPosition(kStartKey, {move1, move2}));
  Check("extended position retains descendant", tree.GetCurrentHead() == node2);
  Check("extended position retains visits",
        node2->move_count.load(std::memory_order_relaxed) == 37);
  Check("extended position retains value",
        node2->win.load(std::memory_order_relaxed) == 23.0f);
  Check("one-child start array is not reallocated",
        start->child.get() == start_children &&
            start->child_nodes.get() == start_child_nodes);
  Check("one-child history array is not reallocated",
        node1->child.get() == node1_children &&
            node1->child_nodes.get() == node1_child_nodes);

  Check("identical position reuses tree",
        tree.ResetToPosition(kStartKey, {move1, move2}));
  Check("identical position retains root", tree.GetCurrentHead() == node2);
  Check("identical traversal remains allocation-free",
        start->child.get() == start_children &&
            start->child_nodes.get() == start_child_nodes &&
            node1->child.get() == node1_children &&
            node1->child_nodes.get() == node1_child_nodes);

  auto* node3 = AppendChild(node2, move3, 19);
  Check("later extension reuses current root",
        tree.ResetToPosition(kStartKey, {move1, move2, move3}));
  Check("later extension retains searched child",
        tree.GetCurrentHead() == node3 &&
            node3->move_count.load(std::memory_order_relaxed) == 19);
}

void TestMultiChildPruning() {
  constexpr uint64_t kStartKey = 0x1a2b;
  NodeTree tree;
  tree.ResetToPosition(kStartKey, {});

  ShogiBoard board;
  board.SetStartPos();
  auto* start = tree.GetCurrentHead();
  start->ExpandNode(&board);
  Check("start position has multiple children", start->child_num > 1);
  if (start->child_num <= 1) return;

  const int selected_index = 1;
  const Move selected_move = start->child[selected_index].move;
  start->child[selected_index].nnrate = 0.375f;
  start->child[selected_index].move_count.store(29,
                                                std::memory_order_relaxed);
  auto* selected = start->CreateChildNode(selected_index);
  selected->move_count.store(17, std::memory_order_relaxed);
  start->CreateChildNode(0)->move_count.store(13, std::memory_order_relaxed);

  Check("multi-child position reuses selected subtree",
        tree.ResetToPosition(kStartKey, {selected_move}));
  Check("selected child becomes current root",
        tree.GetCurrentHead() == selected);
  Check("unused siblings are compacted", start->child_num == 1);
  Check("selected edge move is retained",
        start->child[0].move == selected_move);
  Check("selected edge statistics are retained",
        start->child[0].nnrate == 0.375f &&
            start->child[0].move_count.load(std::memory_order_relaxed) == 29);
  Check("selected node statistics are retained",
        selected->move_count.load(std::memory_order_relaxed) == 17);
}

void TestRewindResetsHead() {
  constexpr uint64_t kStartKey = 0x2345;
  const Move move1 = Move::Parse("7g7f");
  const Move move2 = Move::Parse("3c3d");

  NodeTree tree;
  tree.ResetToPosition(kStartKey, {});
  auto* start = tree.GetCurrentHead();
  auto* node1 = AppendChild(start, move1, 10);
  AppendChild(node1, move2, 20);
  tree.ResetToPosition(kStartKey, {move1, move2});

  Check("rewind is not reported as reuse",
        !tree.ResetToPosition(kStartKey, {move1}));
  auto* rewound = tree.GetCurrentHead();
  Check("rewound head is fresh", !rewound->IsEvaled());
  Check("rewound head has no stale children", rewound->child_num == 0);
}

void TestDivergenceResetsHead() {
  constexpr uint64_t kStartKey = 0x3456;
  const Move old1 = Move::Parse("7g7f");
  const Move old2 = Move::Parse("3c3d");
  const Move new1 = Move::Parse("2g2f");
  const Move new2 = Move::Parse("8c8d");

  NodeTree tree;
  tree.ResetToPosition(kStartKey, {});
  auto* old_node1 = AppendChild(tree.GetCurrentHead(), old1, 10);
  AppendChild(old_node1, old2, 20);
  tree.ResetToPosition(kStartKey, {old1, old2});

  Check("divergent history is not reported as reuse",
        !tree.ResetToPosition(kStartKey, {new1, new2}));
  auto* divergent = tree.GetCurrentHead();
  Check("divergent head is fresh", !divergent->IsEvaled());
  Check("divergent head has no stale children", divergent->child_num == 0);
}

void TestChangedStartingPosition() {
  const Move move = Move::Parse("7g7f");
  NodeTree tree;
  tree.ResetToPosition(0x4567, {});
  AppendChild(tree.GetCurrentHead(), move, 42);
  tree.ResetToPosition(0x4567, {move});

  Check("changed starting position is not reused",
        !tree.ResetToPosition(0x5678, {}));
  Check("changed starting position gets fresh root",
        !tree.GetCurrentHead()->IsEvaled() &&
            tree.GetCurrentHead()->child_num == 0);
}

void TestExplicitNewGameReset() {
  constexpr uint64_t kStartKey = 0x6789;
  const Move move = Move::Parse("7g7f");
  NodeTree tree;
  tree.ResetToPosition(kStartKey, {});
  auto* old_child = AppendChild(tree.GetCurrentHead(), move, 42);
  old_child->win.store(17.0f, std::memory_order_relaxed);
  tree.ResetToPosition(kStartKey, {move});

  tree.DeallocateTree();

  Check("explicit new-game position is not reused",
        !tree.ResetToPosition(kStartKey, {move}));
  auto* fresh = tree.GetCurrentHead();
  Check("explicit new-game root has no stale visits", !fresh->IsEvaled());
  Check("explicit new-game root has no stale children",
        fresh->child_num == 0);
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();

  TestExtensionAndIdenticalPosition();
  TestMultiChildPruning();
  TestRewindResetsHead();
  TestDivergenceResetsHead();
  TestChangedStartingPosition();
  TestExplicitNewGameReset();

  std::printf("\n=== Tree reuse: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
