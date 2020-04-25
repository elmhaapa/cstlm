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
    // std::cout << "SLKN word_vec size: " << word_vec.size() << std::endl;
    // using clock = std::chrono::high_resolution_clock;

    using node_type = typename t_idx::cst_type::node_type;
    using value_type = typename t_idx::value_type;
    typedef std::vector<value_type> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;


    if (ismkn) {
        double final_score = 0;
        // uint64_t size = std::pow(word_vec.size(), 2);
        uint64_t size = word_vec.size() * ngramsize * 2;
        // std::cout << "size: " << size << std::endl;
        
        std::vector<node_type> node_incl_buf;
        node_incl_buf.resize(size);

        std::vector<node_type> node_excl_buf;
        node_excl_buf.resize(size);

        std::vector<uint64_t> start_idx;
        start_idx.resize(size);

        std::vector<uint64_t> end_idx;
        end_idx.resize(size);

        std::vector<size_t> sizes;
        sizes.resize(size);

        std::vector<bool> oks;
        oks.resize(size);

        std::vector<bool> breaks;
        breaks.resize(size);

        std::vector<bool> cont;
        cont.resize(size);

        std::vector<uint8_t> idxs;
        idxs.resize(size);


        LMQueryMKN<t_idx, t_pattern> query(&idx, ngramsize);
        // auto append_start = clock::now();
        
        // std::cout << "enter append symbol" << std::endl;
        query.append_symbol(word_vec, start_idx, end_idx, node_incl_buf, node_excl_buf, sizes, oks, breaks, cont, idxs);

        // std::cout << "query.node_step: " << query.node_step << std::endl;
        // auto append_end = clock::now();

        // auto reverse_creation_start = clock::now();
        std::vector<uint32_t> reverse;
        reverse.resize(size);
        for (auto i = 0; i < size; ++i) {
          reverse[i] = i;
        }
        // auto reverse_creation_end = clock::now();
/*
        node_type node_incl_buf_sorted[size];
        std::copy(node_incl_buf, node_incl_buf + query.node_step, node_incl_buf_sorted);

        node_type node_excl_buf_sorted[size];
        std::copy(node_excl_buf, node_excl_buf + query.node_step, node_excl_buf_sorted);

        uint64_t start_idx_sorted[size];
        std::copy(start_idx, start_idx  + query.node_step, start_idx_sorted);

        uint64_t end_idx_sorted[size];
        std::copy(end_idx, end_idx  + query.node_step, end_idx_sorted);

        uint8_t idxs_sorted[size];
        std::copy(idxs, idxs  + query.node_step, idxs_sorted);
       
        bool oks_sorted[size];
        std::copy(oks, oks  + query.node_step, oks_sorted);
*/

        // We need breaks, cont, ands sizes reverted.
        // So instead of reversing them later we just copy new ones for sorting.
        /*
        bool breaks_sorted[size];
        std::copy(breaks, breaks  + query.node_step, breaks_sorted);

        bool cont_sorted[size];
        std::copy(cont, cont  + query.node_step, cont_sorted);
  */
        // auto creating_sizes_sorted_start = clock::now();
        std::vector<size_t> sizes_sorted(sizes.begin(), sizes.begin() + query.node_step);
        //std::copy(sizes, sizes  + query.node_step, sizes_sorted);
        // auto creating_sizes_sorted_end = clock::now();


        // auto sort_start = clock::now();
        /*
        quicksort<t_idx>(
            idx, 
            reverse,
            node_incl_buf, 
            node_excl_buf,
            start_idx,
            end_idx,
            sizes_sorted,
            oks,
            idxs,
            0, 
            query.node_step -1
          );
          */
        std::stable_sort(reverse.begin(), reverse.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});
 
        std::stable_sort(start_idx.begin(), start_idx.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});
 
        std::stable_sort(end_idx.begin(), end_idx.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});
 
        std::stable_sort(sizes_sorted.begin(), sizes_sorted.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});

        std::stable_sort(oks.begin(), oks.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});

        std::stable_sort(idxs.begin(), idxs.begin() + query.node_step,
            [&node_incl_buf, &idx](size_t i1, size_t i2) {return idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1])) < idx.precomputed.m_bv_rank(idx.cst.id(node_incl_buf[i1]));});

        // auto sort_end = clock::now();

        // std::cout << "passed sorting" << std::endl;

        std::vector<double> cs;
        cs.resize(size);
        std::vector<double> gammas;
        gammas.resize(size);
        std::vector<double> ds;
        ds.resize(size);

        // std::cout << "passed cs, gammas, ds" << std::endl;

        // auto compute_start = clock::now();
        query.compute(word_vec, start_idx, end_idx, node_incl_buf, node_excl_buf, sizes_sorted, oks, cs, gammas, ds, idxs);
        // auto compute_end = clock::now();

        // auto reverse_start = clock::now();
        
        // std::cout << "passed compute" << std::endl;

        std::vector<double> finalcs;
        finalcs.resize(query.node_step);
        std::vector<double> finalgammas;
        finalgammas.resize(query.node_step);
        std::vector<double> finalds;
        finalds.resize(query.node_step);

        //size_t finalsizes[size];
        for (auto i = 0; i < query.node_step; ++i) {
          finalcs[reverse[i]] = cs[i];
          finalgammas[reverse[i]] = gammas[i];
          finalds[reverse[i]] = ds[i];
          //finalsizes[reverse[i]] = sizes_sorted[i];
        }
        // auto reverse_end = clock::now();

        // std::cout << "passed reverse" << std::endl;

        // auto finale_start = clock::now();
        double finalScore = query.finale(sizes, breaks, cont, finalcs, finalgammas, finalds);
        // auto finale_end = clock::now();

        // std::cout << "MY RESULT: " << finalScore << std::endl;
        // LOG(INFO) << "sentence_logprob_kneser_ney for: "
        // << idx.m_vocab.id2token(word_vec.begin(), word_vec.end())
        // << " returning: " << final_score;
        // std::cout << "final score: " << final_score << std::endl;
        /*
        if (finalScore != final_score) {
          std::cout << "NO MATCH!" << std::endl;
          std::cout << "MY RESULT: " << finalScore << std::endl;
          std::cout << "final score: " << final_score << std::endl;
        }
        */

        /*
        LOG(INFO) << "Append = " << duration_cast<microseconds>(append_end - append_start).count() / 1000.0f << " ms";

        LOG(INFO) << "Reverse creation = " << duration_cast<microseconds>(reverse_creation_end - reverse_creation_start).count() / 1000.0f << " ms";

        LOG(INFO) << "Creating sizes sorted = " << duration_cast<microseconds>(creating_sizes_sorted_end - creating_sizes_sorted_start).count() / 1000.0f << " ms";


        LOG(INFO) << "Sort = " << duration_cast<microseconds>(sort_end - sort_start).count() / 1000.0f << " ms";


        LOG(INFO) << "Compute = " << duration_cast<microseconds>(compute_end - compute_start).count() / 1000.0f << " ms";


        LOG(INFO) << "Reverse = " << duration_cast<microseconds>(reverse_end - reverse_start).count() / 1000.0f << " ms";


        LOG(INFO) << "Finale = " << duration_cast<microseconds>(finale_end - finale_start).count() / 1000.0f << " ms";

        */
        return finalScore;
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
