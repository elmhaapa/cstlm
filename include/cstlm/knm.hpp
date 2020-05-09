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
#include "step.h"
#include "computeresult.h"

namespace cstlm {
 

template <class t_idx, class t_pattern>
double sentence_logprob_kneser_ney(const t_idx& idx, const t_pattern& word_vec,
    uint64_t& , uint64_t ngramsize,
    bool ismkn)
{
    // std::cout << "SLKN word_vec size: " << word_vec.size() << std::endl;
    // using clock = std::chrono::high_resolution_clock;

    using node_type = typename t_idx::cst_type::node_type;

    if (ismkn) {
        // uint64_t size = std::pow(word_vec.size(), 2);
        uint64_t size = word_vec.size() * ngramsize * 2;
        // std::cout << "size: " << size << std::endl;

        std::vector<Step<node_type>> steps;
        steps.resize(size);

        LMQueryMKN<t_idx, t_pattern> query(&idx, ngramsize);
        query.append_symbol(word_vec, steps);
        steps.resize(query.node_step);

        std::vector<ComputeResult> cr;
        cr.resize(query.node_step);

        std::sort(steps.begin(), steps.end(),
          [&](Step<node_type> a, Step<node_type> b){ return idx.cst.id(a.node_incl) < idx.cst.id(b.node_incl); });

        // auto compute_start = clock::now();
        query.compute(word_vec, steps, cr);
        // auto compute_end = clock::now();
        
        /*
        std::vector<Step<node_type>> nsteps;
        nsteps.resize(query.node_step);
        */
        std::vector<ComputeResult> ncr;
        ncr.resize(query.node_step);
        for (uint64_t i = 0; i < query.node_step; ++i) {
          // nsteps[steps[i].node_step] = steps[i];
          ncr[steps[i].node_step] = cr[i];
        }
        
        double finalScore = query.finale(ncr);


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
