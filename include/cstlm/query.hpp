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
    double append_symbol(
        const t_pattern& word_vec,
        const value_type& symbol,
        uint64_t* start_idx,
        uint64_t* end_idx,
        node_type* node_incl_buf,
        node_type* node_excl_buf,
        size_t* sizes,
        bool* oks,
        bool* breaks,
        bool* cont,
        uint8_t* idxs
    );
/*
    void compute(
      uint64_t* start_idx,
      uint64_t* end_idx,
      node_type* node_incl_buf,
      node_type* node_excl_buf,
      size_t* sizes,
      bool* oks,
      double* ps
    );
*/
    void compute2(
      const t_pattern& word_vec,
      uint64_t* start_idx,
      uint64_t* end_idx,
      node_type* node_incl_buf,
      node_type* node_excl_buf,
      size_t* sizes,
      bool* oks,
      bool* breaks,
      bool* cont,
      double* cs,
      double* gammas,
      double* ds,
      uint64_t thesize,
      uint8_t* idxs
    );

    double finale(
      size_t* sizes,
      bool* breaks,
      bool* cont,
      double* cs,
      double* gammas,
      double* ds
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
void LMQueryMKN<t_idx, t_pattern>::compute2(
      const t_pattern& word_vec,
      uint64_t* start_idx,
      uint64_t* end_idx,
      node_type* node_incl_buf,
      node_type* node_excl_buf,
      size_t* sizes,
      bool* oks,
      bool* breaks,
      bool* cont,
      double* cs,
      double* gammas,
      double* ds,
      uint64_t thesize,
      uint8_t* idxs
    ) {

  std::cout << "COMPUTE2" << std::endl;
  for (auto a = 0; a < thesize; ++a) {
      auto i = idxs[a];
      auto size = sizes[a];

      double p = 1.0;
      auto node_incl = node_incl_buf[a];

      // std::cout << "node_incl: " << node_incl << std::endl;
      auto node_excl = node_excl_buf[a];

      // std::cout << "node_excl: " << node_incl << std::endl;
      auto ok = oks[a];

      // std::cout << "ok: " << ok << std::endl;
      auto start = word_vec[start_idx[a]];

      // std::cout << "start: " << *start << std::endl;
      auto pattern_end = word_vec[end_idx[a]];

      // std::cout << "CATCH" << std::endl;
      double D1, D2, D3p;
      m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

      double c, d;
      /*
      std::cout << "---" << std::endl;
      std::cout << "i: " << i << std::endl;
      std::cout << "ok: " << ok << std::endl;
      std::cout << "size: " << size << std::endl;
      std::cout << "cst.size(node_incl): " << m_idx->cst.size(node_incl) << std::endl;
      std::cout << "cst.size(node_excl): " << m_idx->cst.size(node_excl) << std::endl;
      std::cout << "---" << std::endl;
      */
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
          d = m_idx->m_N1PlusFrontBack(node_excl, start, pattern_end, size);
         // std::cout << "the d: " << d << std::endl;
      }

      // std::cout << "c: " << c << " d: " << d << std::endl;
      if (c == 1) {
          c -= D1;
      }
      else if (c == 2) {
          c -= D2;
      }
      else if (c >= 3) {
          c -= D3p;
      }

      // std::cout << "c: " << c << " d: " << d << std::endl;
      uint64_t n1 = 0, n2 = 0, n3p = 0;
      if ((i == m_ngramsize && m_ngramsize != 1) || (start == PAT_START_SYM)) {
          m_idx->m_N123PlusFront(node_excl, start, pattern_end, n1, n2, n3p, size);
          std::cout << "m_N123PlusFront n1: " << n1 << std::endl;
      }
      else if (i == 1 || m_ngramsize == 1) {
          n1 = m_idx->discounts.n1_cnt[1];
          n2 = m_idx->discounts.n2_cnt[1];
          n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
      }
      else {
          m_idx->m_N123PlusFrontPrime(node_excl, start, pattern_end, n1, n2, n3p, size);
          std::cout << "m_N123PlusFrontPrime n1: " << n1 << std::endl;
      }

      // n3p is dodgy
      std::cout << "D1: " << D1 << " n1: " << n1 << " D2: " << " n2: " << n2 << " D3p: " << D3p << " n3p: " << n3p << std::endl;
      double gamma = D1 * n1 + D2 * n2 + D3p * n3p;

     // std::cout << "c: " << c << " gamma: " << gamma << " p: " << p << " d: " << d << std::endl;
      p = (c + gamma * p) / d;
      cs[a] = c;
      gammas[a] = gamma;
      ds[a] = d;     
      // p = (cs[counter] + gammas[counter] * p) / ds[counter];
  }

}
  /*
  for (auto s = 0; s < step; ++s) {
    auto size = sizes[counter];
    if (cont[counter]) {
      psum += log10(1);
      counter++;
      continue;
    }
    double p = 1.0 / (m_idx->vocab.size() - 4);
    for (auto a = 1; a <= size; ++a) {
      counter++;
      auto i = idxs[counter];
      auto size = sizes[counter];
      if (breaks[counter]) {
        break;
      }
      double p = 1.0;
      auto node_incl = node_incl_buf[counter];

      // std::cout << "node_incl: " << node_incl << std::endl;
      auto node_excl = node_excl_buf[counter];

      // std::cout << "node_excl: " << node_incl << std::endl;
      auto ok = oks[counter];

      // std::cout << "ok: " << ok << std::endl;
      auto start = word_vec[start_idx[counter]];

      // std::cout << "start: " << *start << std::endl;
      auto pattern_end = word_vec[end_idx[counter]];

      // std::cout << "CATCH" << std::endl;
      double D1, D2, D3p;
      m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

      double c, d;
      //
      std::cout << "---" << std::endl;
      std::cout << "i: " << i << std::endl;
      std::cout << "ok: " << ok << std::endl;
      std::cout << "size: " << size << std::endl;
      std::cout << "cst.size(node_incl): " << m_idx->cst.size(node_incl) << std::endl;
      std::cout << "cst.size(node_excl): " << m_idx->cst.size(node_excl) << std::endl;
      std::cout << "---" << std::endl;
      //
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
          d = m_idx->m_N1PlusFrontBack(node_excl, start, pattern_end, size);
         // std::cout << "the d: " << d << std::endl;
      }

      // std::cout << "c: " << c << " d: " << d << std::endl;
      if (c == 1) {
          c -= D1;
      }
      else if (c == 2) {
          c -= D2;
      }
      else if (c >= 3) {
          c -= D3p;
      }

      // std::cout << "c: " << c << " d: " << d << std::endl;
      uint64_t n1 = 0, n2 = 0, n3p = 0;
      if ((i == m_ngramsize && m_ngramsize != 1) || (start == PAT_START_SYM)) {
          m_idx->m_N123PlusFront(node_excl, start, pattern_end, n1, n2, n3p, size);
          std::cout << "m_N123PlusFront n1: " << n1 << std::endl;
      }
      else if (i == 1 || m_ngramsize == 1) {
          n1 = m_idx->discounts.n1_cnt[1];
          n2 = m_idx->discounts.n2_cnt[1];
          n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
      }
      else {
          m_idx->m_N123PlusFrontPrime(node_excl, start, pattern_end, n1, n2, n3p, size);
          std::cout << "m_N123PlusFrontPrime n1: " << n1 << std::endl;
      }

      // n3p is dodgy
      std::cout << "D1: " << D1 << " n1: " << n1 << " D2: " << " n2: " << n2 << " D3p: " << D3p << " n3p: " << n3p << std::endl;
      double gamma = D1 * n1 + D2 * n2 + D3p * n3p;

     // std::cout << "c: " << c << " gamma: " << gamma << " p: " << p << " d: " << d << std::endl;
      p = (c + gamma * p) / d;
      cs[counter] = c;
      gammas[counter] = gamma;
      ds[counter] = d;     
      // p = (cs[counter] + gammas[counter] * p) / ds[counter];
*/
    // psum += log10(p);
    // counter++
  // return psum;
  /*
  std::cout << "COMPUTE2" << std::endl;
  for (auto i = 0; i < thesize; ++i) {
        if (sizes[i] == 0) {
          continue;
        }
        // std::cout << "ps[i]: " << ps[i] << std::endl;

        auto size = sizes[i];
        double p = 1.0;
        auto node_incl = node_incl_buf[i];

        // std::cout << "node_incl: " << node_incl << std::endl;
        auto node_excl = node_excl_buf[i];

        // std::cout << "node_excl: " << node_incl << std::endl;
        auto ok = oks[i];

        // std::cout << "ok: " << ok << std::endl;
        auto start = word_vec[start_idx[i]];

        // std::cout << "start: " << *start << std::endl;
        auto pattern_end = word_vec[end_idx[i]];

        // std::cout << "CATCH" << std::endl;
        double D1, D2, D3p;
        m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

        double c, d;
        std::cout << "ok: " << ok << std::endl;
        std::cout << "size: " << size << std::endl;
        std::cout << "cst.size(node_incl): " << m_idx->cst.size(node_incl) << std::endl;
        std::cout << "cst.size(node_excl): " << m_idx->cst.size(node_excl) << std::endl;
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
            d = m_idx->m_N1PlusFrontBack(node_excl, start, pattern_end, size);
            std::cout << "the d: " << d << std::endl;
        }

        std::cout << "c: " << c << " d: " << d << std::endl;
        if (c == 1) {
            c -= D1;
        }
        else if (c == 2) {
            c -= D2;
        }
        else if (c >= 3) {
            c -= D3p;
        }

        std::cout << "c: " << c << " d: " << d << std::endl;
        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == m_ngramsize && m_ngramsize != 1) || (start == PAT_START_SYM)) {
            m_idx->m_N123PlusFront(node_excl, start, pattern_end, n1, n2, n3p, size);
        }
        else if (i == 1 || m_ngramsize == 1) {
            n1 = m_idx->discounts.n1_cnt[1];
            n2 = m_idx->discounts.n2_cnt[1];
            n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
        }
        else {
            m_idx->m_N123PlusFrontPrime(node_excl, start, pattern_end, n1, n2, n3p, size);
        }

        // n3p is dodgy
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
        cs[i] = c;
        gammas[i] = gamma;
        ds[i] = d;
        std::cout << "c: " << c << " gamme: " << gamma << " d: " << d << std::endl;
  }
  */

/*
template <class t_idx, class t_pattern>
void LMQueryMKN<t_idx, t_pattern>::compute(
      uint64_t* start_idx,
      uint64_t* end_idx,
      node_type* node_incl_buf,
      node_type* node_excl_buf,
      size_t* sizes,
      bool* oks,
      double* ps
    ) {
  std::cout << "UUS" << std::endl;
  double psum = 0;
  auto counter = 0;
  for (auto s = 0; s < step; ++s) {
    double p = 1.0 / (m_idx->vocab.size() - 4);
    auto size = sizes[counter];
    if (size == 0) {
      psum += log10(1);
      counter++;
      continue;
    }

    for (auto i = 1; i <= size; ++i) {
      counter++;
      // std::cout << "  " << counter << ":" << std::endl;
      auto start = start_idx[counter];
      auto pattern_end = end_idx[counter];
      auto size = sizes[counter];
      if (size == 0) {
        // std::cout << "  in break! " << start << " " << end << std::endl;
        break;
      }
     //  std::cout << "incl: " << node_incl_buf[counter] << std::endl;
     //  std::cout << "excl: " << node_excl_buf[counter] << std::endl;



        auto node_incl = node_incl_buf[counter];
        auto node_excl = node_excl_buf[counter];
        auto ok = oks[counter];

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
        }
        else if (i == 1 || m_ngramsize == 1) {
            n1 = m_idx->discounts.n1_cnt[1];
            n2 = m_idx->discounts.n2_cnt[1];
            n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
        }
        else {
            m_idx->N123PlusFrontPrime(node_excl, start, pattern_end - 1, n1, n2, n3p);
        }

        // n3p is dodgy
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
        ps[counter] = p;
    }
    psum += log10(p);
    counter++;
  }
 // return psum;
}
*/
template <class t_idx, class t_pattern>
double LMQueryMKN<t_idx, t_pattern>::finale(
      size_t* sizes,
      bool* breaks,
      bool* cont,
      double* cs,
      double* gammas,
      double* ds
    ) {
  std::cout << "FINAL" << std::endl;
/*  
  for (auto i = 0; i < node_step; ++i) {
      std::cout << sizes[i] << " ";
  }
  std::cout << std::endl << "----" << std::endl;
  for (auto i = 0; i < node_step; ++i) {
      std::cout << breaks[i] << " ";
  }
  std::cout << std::endl << "----" << std::endl;
  std::cout << std::endl << "----" << std::endl;
  for (auto i = 0; i < node_step; ++i) {
      std::cout << cont[i] << " ";
  }
  std::cout << std::endl << "----" << std::endl;
 */
  double psum = 0.0;
  auto counter = 0;
  for (auto s = 0; s < step; ++s) {
    auto size = sizes[counter];
    if (cont[counter]) {
      std::cout << "case 1" << std::endl;
      std::cout << "psum: " << psum << " adding: " << log10(1) << std::endl;
      psum += log10(1);
      counter++;
      continue;
    }
    double p = 1.0 / (m_idx->vocab.size() - 4);
    for (auto i = 1; i <= size; ++i) {
      counter++;
      if (breaks[counter]) {
        break;
      }

      std::cout << "c: " << cs[counter] << " gamma: " << gammas[counter] << " p: " << p << " d: " << ds[counter] << std::endl;
      p = (cs[counter] + gammas[counter] * p) / ds[counter];

    }


    std::cout << "case 2" << std::endl;
    std::cout << "psum: " << psum << " adding: " << log10(p) << std::endl;
    psum += log10(p);
    counter++;
  }
  return psum;
}

template <class t_idx, class t_pattern>
double LMQueryMKN<t_idx, t_pattern>::append_symbol(
    const t_pattern& word_vec,
    const value_type& symbol,
    uint64_t* start_idx,
    uint64_t* end_idx,
    node_type* node_incl_buf,
    node_type* node_excl_buf,
    size_t* sizes,
    bool* oks,
    bool* breaks,
    bool* cont,
    uint8_t* idxs
    )
{
    
    if (symbol == PAT_START_SYM && m_pattern.size() == 1 && m_pattern.front() == PAT_START_SYM) {
        cont[node_step] = true;
        step++;
        node_step++;
        std::cout << "case 1" << std::endl;
        return log10(1);
    }

    m_pattern_end++;
    m_pattern.push_back(symbol);
    while (m_ngramsize > 0 && m_pattern.size() > m_ngramsize) {
        m_pattern.pop_front();
        m_pattern_start++;
    }
    std::vector<value_type> pattern(m_pattern.begin(), m_pattern.end());

    // fast way, tracking state
    double p = 1.0 / (m_idx->vocab.size() - 4);
    node_type node_incl = m_idx->cst.root(); // v_F^all matching the full pattern, including last item
    auto node_excl_it = m_last_nodes_incl.begin(); // v_F     matching only the context, excluding last item
    node_type node_excl = *node_excl_it;
    auto pattern_begin = pattern.begin();
    auto pattern_end = pattern.end();

    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end - 1) == UNKNOWN_SYM);
    bool ok = !unk;
    std::vector<node_type> node_incl_vec({ node_incl });

    sizes[node_step] = size;

    /*
    std::cout << "size: " << size << std::endl;
    std::cout << "pattern_begin: " << *pattern.begin() << std::endl;
    std::cout << "pattern_end: " << *pattern.end() << std::endl;
    std::cout << "----" << std::endl;
    */
    for (unsigned i = 1; i <= size; ++i) {
        node_step++;
        idxs[node_step] = i;
        auto start = pattern_end - i;

        if (i > 1 && *start == UNKNOWN_SYM) {
          // std::cout << "  " << node_step << " in break!" << std::endl;  
          breaks[node_step] = true;
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
                // std::cout << "  " << node_step << " in break!" << std::endl;  
                // std::cout << "whaat: " << start_idx[node_step] << std::endl;
                breaks[node_step] = true;
                break;
            }
            else {
                node_excl = *node_excl_it;
            }
        }

        // std::cout << "  start: " << *start << std::endl;
        // std::cout << "  pattern_end: " << *pattern_end << std::endl;
        // std::cout << "start: " << *start << " pattern_end: " << *pattern_end << std::endl;
        // std::cout << "  incl: " << node_incl << std::endl;
        // std::cout << "  excl: " << node_excl << std::endl;
        node_incl_buf[node_step] = node_incl;
        node_excl_buf[node_step] = node_excl;
        //size_t n_size = sizes[node_step];
        sizes[node_step] = std::distance(start, pattern_end - 1);
        start_idx[node_step] = m_pattern_end - i;
        end_idx[node_step] = m_pattern_end - 1;
        oks[node_step] = ok;
        /*
        std::cout << "---" << std::endl;
        std::cout << "i: " << i << std::endl;
        std::cout << "ok: " << ok << std::endl;
        std::cout << "n_size: " << n_size << std::endl;
        std::cout << "cst.size(node_incl): " << m_idx->cst.size(node_incl) << std::endl;
        std::cout << "cst.size(node_excl): " << m_idx->cst.size(node_excl) << std::endl;
        std::cout << "---" << std::endl;
*/
        // std::cout << "start: " << *start << " my: " << word_vec[m_pattern_end - i] << std::endl;
        // std::cout << "pattern_end: " << *(pattern_end-1) << " my: " << word_vec[(m_pattern_end-1)] << std::endl;
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
            // std::cout << "the d: " << d << std::endl;
        }
        // std::cout << "c: " << c << " d: " << d << std::endl;
        if (c == 1) {
            c -= D1;
        }
        else if (c == 2) {
            c -= D2;
        }
        else if (c >= 3) {
            c -= D3p;
        }

        // std::cout << "c: " << c << " d: " << d << std::endl;
        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            m_idx->N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
          std::cout << "N123PlusFront n1: " << n1 << std::endl;
        }
        else if (i == 1 || m_ngramsize == 1) {
            n1 = m_idx->discounts.n1_cnt[1];
            n2 = m_idx->discounts.n2_cnt[1];
            n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
        }
        else {
            m_idx->N123PlusFrontPrime(node_excl, start, pattern_end - 1, n1, n2, n3p);
          std::cout << "N123PlusFrontPrime n1: " << n1 << std::endl;
        }

        // n3p is dodgy
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;

      std::cout << "D1: " << D1 << " n1: " << n1 << " D2: " << " n2: " << n2 << " D3p: " << D3p << " n3p: " << n3p << std::endl;
        // std::cout << "c: " << c << " gamma: " << gamma << " p: " << p << " d: " << d << std::endl;
        p = (c + gamma * p) / d;
        //LOG(INFO) << "\t\ti = " << i << " p = " << p << " c = " << c << " gamma " << gamma << " d = " << d;
        //LOG(INFO) << "\t\t\t" << D1 << ":" << n1 << ":" << D2 << ":" << n2 << ":" << D3p << ":" << n3p;
    }
//    std::cout << "----" << std::endl;

    m_last_nodes_incl = node_incl_vec;
    while (m_pattern.size() > m_last_nodes_incl.size()) {
        m_pattern.pop_front();
         m_pattern_start++;
    }

    step++;
    node_step++;
    std::cout << "case 2" << std::endl;
    return log10(p);
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
