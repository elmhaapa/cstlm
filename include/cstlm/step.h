#ifndef STEP_H
#define STEP_H

template<class t_node>
struct Step
{
  t_node node_incl;
  t_node node_excl;
  uint64_t start_idx;
  uint64_t end_idx;
  size_t size;
  bool ok;
  bool brk;
  bool cont;
  uint8_t idx;
  uint64_t node_step;
};

#endif
