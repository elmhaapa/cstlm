#pragma once

#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <iomanip>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"
#include "constants.hpp"
#include "step.h"
#include "computeresult.h"

namespace cstlm {

// Returns the Kneser-Ney probability of a sentence, word at a
// time. Words are supplied using the append_symbol method which
// returns the conditional probability of that word given all
// previous words.

template <class t_idx, class t_pattern>
class LMQueryMKN {
public:
    using value_type = typename t_idx::value_type;
    using node_type = typename t_idx::cst_type::node_type;
    using index_type = t_idx;
    
    typedef std::vector<value_type> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;

    uint64_t node_step = 0;

public:
    LMQueryMKN()
    {
        m_idx = nullptr;
    }
    LMQueryMKN(const index_type* idx, uint64_t ngramsize, bool start_sentence = true);
    void append_symbol(
        const t_pattern& word_vec,
        std::vector<Step<node_type>>& steps
    );

    void compute(
      const t_pattern& word_vec,
      std::vector<Step<node_type>>& steps,
      std::vector<ComputeResult>& cr
    );

    double finale(
      std::vector<Step<node_type>>& steps,
      std::vector<ComputeResult>& cr
    );

    bool operator==(const LMQueryMKN& other) const;

    size_t hash() const;

    bool empty() const
    {
        return m_last_nodes_incl.size() == 1 && m_last_nodes_incl.back() == m_idx->cst.root();
    }

    bool is_start() const
    {
        return m_pattern.size() == 1 && m_pattern.back() == PAT_START_SYM;
    }

private:
    const index_type* m_idx;
    uint64_t m_ngramsize;
    std::vector<node_type> m_last_nodes_incl;
    std::deque<value_type> m_pattern;

    uint64_t m_pattern_start = 0;
    uint64_t m_pattern_end = 0;
    uint64_t step = 0;


};

template <class t_idx, class t_pattern>
LMQueryMKN<t_idx, t_pattern>::LMQueryMKN(const t_idx* idx, uint64_t ngramsize, bool start_sentence)
    : m_idx(idx)
    , m_ngramsize(ngramsize)
{
    auto root = m_idx->cst.root();
    m_last_nodes_incl.push_back(root);
    if (start_sentence) {
        auto node = root;
        auto r = backward_search_wrapper(*m_idx, node, PAT_START_SYM);
        (void)r;
        assert(r >= 0);
        m_last_nodes_incl.push_back(node);
        m_pattern_end++;
        m_pattern.push_back(PAT_START_SYM);
    }
}

template <class t_idx, class t_pattern>
void LMQueryMKN<t_idx, t_pattern>::compute(
      const t_pattern& word_vec,
      std::vector<Step<node_type>>& steps,
      std::vector<ComputeResult>& cr
    ) {

  for (uint64_t a = 0; a < node_step; ++a) {
      Step<node_type> step = steps[a];
      auto ks = step.start_idx;
      auto ke = step.end_idx;
      if (ks > word_vec.size()) {
        continue;
      }
      if (ke > word_vec.size()) {
        continue;
      }
      auto i = step.idx;
      auto size = step.size;
      
      auto node_incl = step.node_incl;
      auto node_excl = step.node_excl;
      auto ok = step.ok;
      auto start = word_vec[step.start_idx];

      double D1, D2, D3p;
      m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

      double c, d;

      if ((i == m_ngramsize && m_ngramsize != 1) || (start == PAT_START_SYM)) {
          c = (ok) ? m_idx->cst.size(node_incl) : 0;
          d = m_idx->cst.size(node_excl);
      }
      else if (i == 1 || m_ngramsize == 1) {
          c = (ok) ? m_idx->m_N1PlusBack(node_incl, start) : 0;
          d = m_idx->discounts.N1plus_dotdot;
      }
      else {
          c = (ok) ? m_idx->m_N1PlusBack(node_incl, start) : 0;
          d = m_idx->m_N1PlusFrontBack(node_excl, start, word_vec[step.end_idx - 1], size);
      }

      if (c == 1) {
          c -= D1;
      }
      else if (c == 2) {
          c -= D2;
      }
      else if (c >= 3) {
          c -= D3p;
      }

      uint64_t n1 = 0, n2 = 0, n3p = 0;
      if ((i == m_ngramsize && m_ngramsize != 1) || (start == PAT_START_SYM)) {
          m_idx->m_N123PlusFront(node_excl, word_vec[step.end_idx - 1], n1, n2, n3p, size);
          // std::cout << "m_N123PlusFront n1: " << n1 << " n2: " << n2 << std::endl;
      }
      else if (i == 1 || m_ngramsize == 1) {
          n1 = m_idx->discounts.n1_cnt[1];
          n2 = m_idx->discounts.n2_cnt[1];
          n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
      }
      else {
          m_idx->m_N123PlusFrontPrime(node_excl, start, n1, n2, n3p, size);
          // std::cout << "m_N123PlusFrontPrime n1: " << n1 << " n2: " << n2 << std::endl;
      }

      // n3p is dodgy
      double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
      // std::cout << "D1: " << D1 << " n1: " << n1 << " D2: " << D2 << " n2: " << n2 << " D3p: " << D3p << " n3p: " << n3p << std::endl;
      cr[a].c = c;
      cr[a].gamma = gamma;
      cr[a].d = d;     
    //  std::cout << "c: " << c << " gamma: " << gamma << " d: " << d << std::endl;
  }
}

template <class t_idx, class t_pattern>
double LMQueryMKN<t_idx, t_pattern>::finale(
      std::vector<Step<node_type>>& steps,
      std::vector<ComputeResult>& cr
    ) {
  double psum = 0.0;
  auto counter = 0;
  // std::cout << "FINALE:" << std::endl;
  for (uint64_t s = 0; s < step; ++s) {
    auto size = steps[counter].size;
    if (steps[counter].cont) {
      psum += log10(1);
      counter++;
      continue;
    }
    double p = 1.0 / (m_idx->vocab.size() - 4);
    for (uint64_t i = 1; i <= size; ++i) {
      counter++;
      if (steps[counter].brk) {
        break;
      }
      // std::cout << "c: " << cs[counter] << " gammas: " << gammas[counter] << " d: " << ds[counter] << std::endl;
      p = (cr[counter].c + cr[counter].gamma * p) / cr[counter].d;
    }

    psum += log10(p);
    counter++;
  }
  return psum;
}

template <class t_idx, class t_pattern>
void LMQueryMKN<t_idx, t_pattern>::append_symbol(
    const t_pattern& word_vec,
    std::vector<Step<node_type>>& steps
  )
{
  for (auto symbol : word_vec) {
    if (symbol == PAT_START_SYM && m_pattern.size() == 1 && m_pattern.front() == PAT_START_SYM) {
        steps[node_step].cont = true;
        step++;
        node_step++;
        steps[node_step].node_step = node_step;
        continue;
    }

    m_pattern_end++;
    m_pattern.push_back(symbol);
    while (m_ngramsize > 0 && m_pattern.size() > m_ngramsize) {
        m_pattern.pop_front();
        m_pattern_start++;
    }
    std::vector<value_type> pattern(m_pattern.begin(), m_pattern.end());

    // fast way, tracking state
    // double p = 1.0 / (m_idx->vocab.size() - 4);
    node_type node_incl = m_idx->cst.root(); // v_F^all matching the full pattern, including last item
    auto node_excl_it = m_last_nodes_incl.begin(); // v_F     matching only the context, excluding last item
    node_type node_excl = *node_excl_it;
    auto pattern_begin = pattern.begin();
    auto pattern_end = pattern.end();

    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end - 1) == UNKNOWN_SYM);
    bool ok = !unk;
    std::vector<node_type> node_incl_vec({ node_incl });

    steps[node_step].size = size;

    for (unsigned i = 1; i <= size; ++i) {
        node_step++;
        steps[node_step].node_step = node_step;
        steps[node_step].idx = i;
        auto start = pattern_end - i;

        if (i > 1 && *start == UNKNOWN_SYM) {
          steps[node_step].brk = true;
          break;
        }
        if (ok) {
            ok = backward_search_wrapper(*m_idx, node_incl, *start);
            if (ok)
                node_incl_vec.push_back(node_incl);
        }

        // recycle the node_incl matches from the last call to append_symbol
        // to serve as the node_excl values
        if (i >= 2) {
            node_excl_it++;
            if (node_excl_it == m_last_nodes_incl.end()) {
                steps[node_step].brk = true;
                break;
            }
            else {
                node_excl = *node_excl_it;
            }
        }

        steps[node_step].node_incl = node_incl;
        steps[node_step].node_excl = node_excl;
        //size_t n_size = sizes[node_step];
        steps[node_step].size = std::distance(start, pattern_end - 1);
        steps[node_step].start_idx = m_pattern_end - i;
        steps[node_step].end_idx = m_pattern_end - 1;
        // std::cout << "pattern_end: " << *(pattern_end -1) << " end_idx[node_step]: " << word_vec[m_pattern_end - 1] << std::endl;
        steps[node_step].ok = ok;

        /*
        double D1, D2, D3p;
        m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

        double c, d;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            c = (ok) ? m_idx->cst.size(node_incl) : 0;
            d = m_idx->cst.size(node_excl);
        }
        else if (i == 1 || m_ngramsize == 1) {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : 0;
            d = m_idx->discounts.N1plus_dotdot;
        }
        else {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : 0;
            d = m_idx->N1PlusFrontBack(node_excl, start, pattern_end - 1);
        }
        if (c == 1) {
            c -= D1;
        }
        else if (c == 2) {
            c -= D2;
        }
        else if (c >= 3) {
            c -= D3p;
        }

        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            m_idx->N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
            // std::cout << "N123PlusFront n1: " << n1 << " n2: " << n2 << std::endl;
        }
        else if (i == 1 || m_ngramsize == 1) {
            n1 = m_idx->discounts.n1_cnt[1];
            n2 = m_idx->discounts.n2_cnt[1];
            n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
        }
        else {
            m_idx->N123PlusFrontPrime(node_excl, start, pattern_end - 1, n1, n2, n3p);
            // std::cout << "N123PlusFrontPrime n1: " << n1 << " n2: " << n2 << std::endl;
        }

        // n3p is dodgy
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;

        // std::cout << "D1: " << D1 << " n1: " << n1 << " D2: " << D2 << " n2: " << n2 << " D3p: " << D3p << " n3p: " << n3p << std::endl;
        p = (c + gamma * p) / d;


        // std::cout << "c: " << c << " gammas: " << gamma << " d: " << d << std::endl;
        //LOG(INFO) << "\t\ti = " << i << " p = " << p << " c = " << c << " gamma " << gamma << " d = " << d;
        //LOG(INFO) << "\t\t\t" << D1 << ":" << n1 << ":" << D2 << ":" << n2 << ":" << D3p << ":" << n3p;
    */

    }

    m_last_nodes_incl = node_incl_vec;
    while (m_pattern.size() > m_last_nodes_incl.size()) {
        m_pattern.pop_front();
         m_pattern_start++;
    }

    step++;
    node_step++;
    steps[node_step].node_step = node_step;
  }
}

template <class t_idx, class t_pattern>
bool LMQueryMKN<t_idx, t_pattern>::operator==(const LMQueryMKN& other) const
{
    if (m_idx != other.m_idx)
        return false;
    if (m_pattern.size() != other.m_pattern.size())
        return false;
    if (m_last_nodes_incl.size() != other.m_last_nodes_incl.size())
        return false;
    for (auto i = 0u; i < m_pattern.size(); ++i) {
        if (m_pattern[i] != other.m_pattern[i])
            return false;
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        if (m_last_nodes_incl[i] != other.m_last_nodes_incl[i])
            return false;
    }
    return true;
}

template <class t_idx, class t_pattern>
std::size_t LMQueryMKN<t_idx, t_pattern>::hash() const
{
    std::size_t seed = 0;
    for (auto& i : m_pattern) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        auto id = m_idx->cst.id(m_last_nodes_incl[i]);
        seed ^= id + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
}
