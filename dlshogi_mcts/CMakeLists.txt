add_library(dlshogi_mcts STATIC
  uct_node.cc
  uct_search.cc
)

target_link_libraries(dlshogi_mcts PUBLIC shogi_core mate_solver mcts)
target_include_directories(dlshogi_mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
