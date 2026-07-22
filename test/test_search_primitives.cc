#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "dlshogi_mcts/search_primitives.h"

namespace {

using dlshogi_mcts::BackupTrajectory;
using dlshogi_mcts::EdgeOutcome;
using dlshogi_mcts::PuctParameters;
using dlshogi_mcts::ResolveTerminalEdge;
using dlshogi_mcts::SelectPuctChild;
using dlshogi_mcts::TryGetProvenEdgeValue;
using dlshogi_mcts::child_node_t;
using dlshogi_mcts::trajectory_t;
using dlshogi_mcts::uct_node_t;

int failures = 0;

void Check(const char* name, bool condition) {
  if (condition) return;
  std::printf("  FAIL  %s\n", name);
  ++failures;
}

bool Near(float actual, float expected) {
  return std::fabs(actual - expected) < 1.0e-5f;
}

void InitNode(uct_node_t* node, int child_num) {
  node->child_num = static_cast<short>(child_num);
  node->child = std::make_unique<child_node_t[]>(child_num);
  node->InitChildNodes();
  node->SetEvaled();
}

void TestTerminalEdgeConvention() {
  child_node_t winning_edge;
  const float win =
      ResolveTerminalEdge(&winning_edge, EdgeOutcome::kWin);
  Check("edge win has value one", Near(win, 1.0f));
  Check("edge win marks opponent loss", winning_edge.IsLose());

  child_node_t losing_edge;
  const float loss =
      ResolveTerminalEdge(&losing_edge, EdgeOutcome::kLoss);
  Check("edge loss has value zero", Near(loss, 0.0f));
  Check("edge loss marks opponent win", losing_edge.IsWin());

  child_node_t drawing_edge;
  const float draw =
      ResolveTerminalEdge(&drawing_edge, EdgeOutcome::kDraw, 0.625f);
  Check("edge draw keeps mover draw value", Near(draw, 0.625f));
  Check("edge draw is proven", drawing_edge.IsDraw());

  float proven = -1.0f;
  Check("proven winning edge is readable",
        TryGetProvenEdgeValue(&winning_edge, 0.5f, &proven) &&
            Near(proven, 1.0f));
  Check("proven losing edge is readable",
        TryGetProvenEdgeValue(&losing_edge, 0.5f, &proven) &&
            Near(proven, 0.0f));
  Check("proven draw uses supplied mover value",
        TryGetProvenEdgeValue(&drawing_edge, 0.375f, &proven) &&
            Near(proven, 0.375f));
}

void TestTerminalBackupAlternatesOncePerEdge() {
  uct_node_t root;
  uct_node_t middle;
  uct_node_t leaf_parent;
  InitNode(&root, 1);
  InitNode(&middle, 1);
  InitNode(&leaf_parent, 1);

  // Production adds one virtual visit to every traversed edge before backup.
  for (uct_node_t* node : {&root, &middle, &leaf_parent}) {
    node->move_count.store(1, std::memory_order_relaxed);
    node->child[0].move_count.store(1, std::memory_order_relaxed);
  }

  const std::vector<trajectory_t> path = {
      {&root, 0}, {&middle, 0}, {&leaf_parent, 0}};
  BackupTrajectory(path, 1.0f, 4.0f);

  Check("last edge receives terminal parent win",
        Near(leaf_parent.child[0].win.load(), 1.0f));
  Check("middle edge receives one perspective flip",
        Near(middle.child[0].win.load(), 0.0f));
  Check("root edge receives two perspective flips",
        Near(root.child[0].win.load(), 1.0f));
  Check("leaf moves-left is stored on last edge",
        Near(leaf_parent.child[0].sum_m.load(), 4.0f));
  Check("moves-left increments toward root",
        Near(middle.child[0].sum_m.load(), 5.0f) &&
            Near(root.child[0].sum_m.load(), 6.0f));
  Check("backup resolves virtual visits exactly once",
        root.child[0].move_count.load() == 1 &&
            middle.child[0].move_count.load() == 1 &&
            leaf_parent.child[0].move_count.load() == 1);
}

void TestInvalidBackupIsContained() {
  uct_node_t root;
  InitNode(&root, 1);
  root.move_count.store(1, std::memory_order_relaxed);
  root.child[0].move_count.store(1, std::memory_order_relaxed);
  const std::vector<trajectory_t> path = {{&root, 0}};

  Check("NaN backup is rejected",
        !BackupTrajectory(path, std::numeric_limits<float>::quiet_NaN()));
  Check("rejected backup leaves parent accumulator unchanged",
        Near(root.win.load(), 0.0f));
  Check("rejected backup leaves child accumulator unchanged",
        Near(root.child[0].win.load(), 0.0f));
}

void TestParentQFpuForUnvisitedChildren() {
  uct_node_t node;
  InitNode(&node, 2);
  node.move_count.store(10, std::memory_order_relaxed);
  node.win.store(8.0f, std::memory_order_relaxed);
  node.visited_nnrate.store(0.25f, std::memory_order_relaxed);

  node.child[0].move_count.store(1, std::memory_order_relaxed);
  node.child[0].win.store(0.4f, std::memory_order_relaxed);
  node.child[0].nnrate = 0.0f;
  node.child[1].nnrate = 0.0f;

  PuctParameters params;
  params.c_init = 0.0f;
  params.c_base = 3.0e30f;  // Makes the exploration term zero here.
  params.fpu_reduction = 0.2f;

  // dlshogi FPU gives the unvisited child parent_q = 0.8 -
  // 0.2*sqrt(0.25) = 0.7, above the visited child's 0.4.
  Check("unvisited child uses reduced parent Q",
        SelectPuctChild(nullptr, &node, params) == 1);
}

void TestSelectedPriorAccumulatesForFpu() {
  uct_node_t node;
  InitNode(&node, 3);
  node.child[0].nnrate = 0.1f;
  node.child[1].nnrate = 0.7f;
  node.child[2].nnrate = 0.2f;

  PuctParameters params;
  const unsigned selected = SelectPuctChild(nullptr, &node, params);
  Check("initial PUCT selects highest prior", selected == 1);
  Check("selected prior is accumulated for FPU",
        Near(node.visited_nnrate.load(), 0.7f));
}

void TestDlshogiExplorationConstant() {
  uct_node_t node;
  InitNode(&node, 2);
  node.move_count.store(1, std::memory_order_relaxed);
  node.child[0].move_count.store(1, std::memory_order_relaxed);
  node.child[0].win.store(0.9f, std::memory_order_relaxed);
  node.child[0].nnrate = 0.0f;
  node.child[1].nnrate = 0.85f;

  PuctParameters params;
  params.c_init = 0.0f;
  params.c_base = 1.0f;

  // dlshogi uses log((sum + c_base + 1) / c_base). With sum=1 this
  // gives log(3), so the unvisited child's score exceeds 0.9. Omitting the
  // +1 would use log(2) and incorrectly keep child 0.
  Check("PUCT exploration constant includes dlshogi's plus one",
        SelectPuctChild(nullptr, &node, params) == 1);
}

void TestProvenStatePropagation() {
  PuctParameters params;

  child_node_t parent_with_winning_reply;
  uct_node_t has_winning_reply;
  InitNode(&has_winning_reply, 2);
  has_winning_reply.child[1].SetLose();
  Check("proven winning reply is selected immediately",
        SelectPuctChild(&parent_with_winning_reply, &has_winning_reply,
                        params) == 1);
  Check("winning current node marks incoming edge as opponent win",
        parent_with_winning_reply.IsWin());

  child_node_t parent_all_losses;
  uct_node_t all_losses;
  InitNode(&all_losses, 2);
  all_losses.child[0].SetWin();
  all_losses.child[1].SetWin();
  SelectPuctChild(&parent_all_losses, &all_losses, params);
  Check("all losing replies mark incoming edge as opponent loss",
        parent_all_losses.IsLose());

  child_node_t parent_forced_draw;
  uct_node_t forced_draw;
  InitNode(&forced_draw, 2);
  forced_draw.child[0].SetWin();
  forced_draw.child[1].SetDraw();
  forced_draw.child[1].nnrate = 0.5f;
  Check("draw is selected over a proven loss",
        SelectPuctChild(&parent_forced_draw, &forced_draw, params) == 1);
  Check("win-or-draw replies propagate draw",
        parent_forced_draw.IsDraw());
}

}  // namespace

int main() {
  TestTerminalEdgeConvention();
  TestTerminalBackupAlternatesOncePerEdge();
  TestInvalidBackupIsContained();
  TestParentQFpuForUnvisitedChildren();
  TestSelectedPriorAccumulatesForFpu();
  TestDlshogiExplorationConstant();
  TestProvenStatePropagation();

  std::printf("\n=== Search primitives: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
