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
#include "query.hpp"
#include "query_kn.hpp"

namespace cstlm {
 
template <class t_idx, class t_nt>
uint64_t partition(
    const t_idx& idx, 
    uint64_t* reverse,
    t_nt* node_incl_buf, 
    t_nt* node_excl_buf,
    uint64_t* start_idx,
    uint64_t* end_idx,
    size_t* sizes_sorted,
    bool* oks_sorted,
    bool* breaks_sorted,
    bool* cont_sorted,
    uint8_t* idxs_sorted,
    uint64_t l, 
    uint64_t h
    ) {
    using namespace std;
    auto x = idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[h])); 
    int i = (l - 1); 
  
    for (int j = l; j <= h - 1; j++) { 
        if (idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[j])) <= x) { 
            i++;
            swap(node_incl_buf[i], node_incl_buf[j]); 
            swap(node_excl_buf[i], node_excl_buf[j]); 
            swap(end_idx[i], end_idx[j]);
            swap(start_idx[i], start_idx[j]);
            swap(sizes_sorted[i], sizes_sorted[j]); 
            swap(oks_sorted[i], oks_sorted[j]); 
            swap(breaks_sorted[i], breaks_sorted[j]); 
            swap(cont_sorted[i], cont_sorted[j]); 
            swap(idxs_sorted[i], idxs_sorted[j]); 

            swap(reverse[i], reverse[j]); 
        } 
    } 

    swap(node_incl_buf[i + 1], node_incl_buf[h]); 
    swap(node_excl_buf[i + 1], node_excl_buf[h]); 
    swap(end_idx[i + 1], end_idx[h]);
    swap(start_idx[i + 1], start_idx[h]);
    swap(sizes_sorted[i + 1], sizes_sorted[h]); 
    swap(oks_sorted[i + 1], oks_sorted[h]); 
    swap(breaks_sorted[i + 1], breaks_sorted[h]); 
    swap(cont_sorted[i + 1], cont_sorted[h]); 
    swap(idxs_sorted[i + 1], idxs_sorted[h]); 


    swap(reverse[i + 1], reverse[h]); 
    return (i + 1); 
}
 
template <class t_idx, class t_nt>
void quicksort(
    const t_idx& idx, 
    uint64_t* reverse,
    t_nt* node_incl_buf, 
    t_nt* node_excl_buf,
    uint64_t* start_idx,
    uint64_t* end_idx,
    size_t* sizes_sorted,
    bool* oks_sorted,
    bool* breaks_sorted,
    bool* cont_sorted,
    uint8_t* idxs_sorted,
    uint64_t l, 
    uint64_t h
    ) {
    uint64_t stack[h - l + 1]; 
  
    uint64_t top = 0; 
  
    stack[++top] = l; 
    stack[++top] = h; 
  
    while (top >= 1) { 
        h = stack[top--]; 
        l = stack[top--]; 
  
        uint64_t p = partition<t_idx, t_nt>(
            idx, 
            reverse,
            node_incl_buf, 
            node_excl_buf, 
            start_idx,
            end_idx,
            sizes_sorted,
            oks_sorted,
            breaks_sorted,
            cont_sorted,
            idxs_sorted,
            l, 
            h
        ); 
  
        if (p - 1 > l) { 
            stack[++top] = l; 
            stack[++top] = p - 1; 
        } 
  
        if (p + 1 < h) { 
            stack[++top] = p + 1; 
            stack[++top] = h; 
        } 
    } 
}

template <class t_idx, class t_pattern>
double sentence_logprob_kneser_ney(const t_idx& idx, const t_pattern& word_vec,
    uint64_t& M, uint64_t ngramsize,
    bool ismkn)
{
    using node_type = typename t_idx::cst_type::node_type;
    using value_type = typename t_idx::value_type;
    typedef std::vector<value_type> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;


    if (ismkn) {
        double final_score = 0;
        uint64_t size = std::pow(word_vec.size(), 2);
        node_type node_incl_buf[size];
        node_type node_excl_buf[size];
        uint64_t start_idx[size] = {0};
        uint64_t end_idx[size] = {0};
        size_t sizes[size] = {0};
        bool oks[size] = {0};
        bool breaks[size] = {0};
        bool cont[size] = {0};
        uint8_t idxs[size] = {0};


        LMQueryMKN<t_idx, t_pattern> query(&idx, size, ngramsize);
        for (const auto& word : word_vec) {
            auto the_score = query.append_symbol(word_vec, word, start_idx, end_idx, node_incl_buf, node_excl_buf, sizes, oks, breaks, cont, idxs);
            final_score += the_score;
            //LOG(INFO) << "\tprob: " << idx.m_vocab.id2token(word) << " is: " << prob;
        }

        uint64_t reverse[size];
        for (auto i = 0; i < size; ++i) {
          reverse[i] = i;
        }

        node_type node_incl_buf_sorted[size];
        std::copy(node_incl_buf, node_incl_buf + query.node_step, node_incl_buf_sorted);

        node_type node_excl_buf_sorted[size];
        std::copy(node_excl_buf, node_excl_buf + query.node_step, node_excl_buf_sorted);

        uint64_t start_idx_sorted[size];
        std::copy(start_idx, start_idx  + query.node_step, start_idx_sorted);

        uint64_t end_idx_sorted[size];
        std::copy(end_idx, end_idx  + query.node_step, end_idx_sorted);

        size_t sizes_sorted[size];
        std::copy(sizes, sizes  + query.node_step, sizes_sorted);

        bool oks_sorted[size];
        std::copy(oks, oks  + query.node_step, oks_sorted);

        bool breaks_sorted[size];
        std::copy(breaks, breaks  + query.node_step, breaks_sorted);

        bool cont_sorted[size];
        std::copy(cont, cont  + query.node_step, cont_sorted);
  
        uint8_t idxs_sorted[size];
        std::copy(idxs, idxs  + query.node_step, idxs_sorted);
       
        quicksort<t_idx>(
            idx, 
            reverse,
            node_incl_buf_sorted, 
            node_excl_buf_sorted,
            start_idx_sorted,
            end_idx_sorted,
            sizes_sorted,
            oks_sorted,
            breaks_sorted,
            cont_sorted,
            idxs_sorted,
            0, 
            query.node_step -1
          );

        double cs[size];
        double gammas[size];
        double ds[size];


        query.compute(word_vec, start_idx_sorted, end_idx_sorted, node_incl_buf_sorted, node_excl_buf_sorted, sizes_sorted, oks_sorted, breaks_sorted, cont_sorted, cs, gammas, ds, size, idxs_sorted);
        double finalcs[size];
        double finalgammas[size];
        double finalds[size];
        size_t finalsizes[size];
        bool finalbreaks[size];
        bool finalcont[size];
        for (auto i = 0; i < size; ++i) {
          finalcs[reverse[i]] = cs[i];
          finalgammas[reverse[i]] = gammas[i];
          finalds[reverse[i]] = ds[i];
          finalsizes[reverse[i]] = sizes_sorted[i];
          finalbreaks[reverse[i]] = breaks_sorted[i];
          finalcont[reverse[i]] = cont_sorted[i];
        }
         double finalScore = query.finale(finalsizes, finalbreaks, finalcont, finalcs, finalgammas, finalds);
        // std::cout << "MY RESULT: " << finalScore << std::endl;
        // LOG(INFO) << "sentence_logprob_kneser_ney for: "
        // << idx.m_vocab.id2token(word_vec.begin(), word_vec.end())
        // << " returning: " << final_score;
        // std::cout << "final score: " << final_score << std::endl;
        if (finalScore != final_score) {
          std::cout << "NO MATCH!" << std::endl;
          std::cout << "MY RESULT: " << finalScore << std::endl;
          std::cout << "final score: " << final_score << std::endl;
        }
        return final_score;
    }
    else {
        double final_score = 0;
        LMQueryKN<t_idx> query(&idx, ngramsize);
        for (const auto& word : word_vec)
            final_score += query.append_symbol(word);
        return final_score;
    }
}

template <class t_idx, class t_pattern>
double sentence_perplexity_kneser_ney(const t_idx& idx, t_pattern& pattern,
    uint32_t ngramsize, bool ismkn)
{
    auto pattern_size = pattern.size();
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M = pattern_size + 1;
    double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, ismkn);
    double perplexity = pow(10, -(1 / (double)M) * sentenceprob);
    return perplexity;
}

// required by Moses
template <class t_idx, class t_pattern>
uint64_t patternId(const t_idx& idx, const t_pattern& word_vec)
{
    uint64_t lb = 0, rb = idx.cst.size() - 1;
    backward_search(idx.cst.csa, lb, rb, word_vec.begin(), word_vec.end(), lb,
        rb);
    auto node = idx.cst.node(lb, rb);
    return idx.cst.id(node);
}
}
