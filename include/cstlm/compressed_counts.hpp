#pragma once

#include <iostream>

#include "sdsl/vectors.hpp"

#include "constants.hpp"
#include "collection.hpp"
#include "logging.hpp"
#include "utils.hpp"

namespace cstlm {

template <class t_bv = sdsl::rrr_vector<15>, class t_vec = sdsl::dac_vector<> >
struct compressed_counts {
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_bv bv_type;
    typedef typename bv_type::rank_1_type rank_type;
    typedef t_vec vector_type;

private:
    bv_type m_bv;
    rank_type m_bv_rank;
    bool m_is_mkn;
    vector_type m_counts;
    uint8_t _fb = 0;
    uint8_t _f1prime = 1;
    uint8_t _f2prime = 2;
    uint8_t _b = 3;
    uint8_t _f1 = 4;
    uint8_t _f2 = 5;
    uint8_t shift = 6;

public:
    compressed_counts() = default;
    compressed_counts(const compressed_counts& cc)
    {
        m_bv = cc.m_bv;
        m_bv_rank = cc.m_bv_rank;
        m_bv_rank.set_vector(&m_bv);
        m_is_mkn = cc.m_is_mkn;
        m_counts = cc.m_counts;
    }
    compressed_counts(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank = std::move(cc.m_bv_rank);
        m_bv_rank.set_vector(&m_bv);
        m_is_mkn = cc.m_is_mkn;
        m_counts = std::move(cc.m_counts);
    }
    compressed_counts& operator=(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank = std::move(cc.m_bv_rank);
        m_bv_rank.set_vector(&m_bv);
        m_is_mkn = cc.m_is_mkn;
        m_counts = std::move(cc.m_counts);
        return *this;
    }

    template <class t_cst>
    compressed_counts(collection& col, t_cst& cst, uint64_t max_node_depth,
        bool mkn_counts)
    {
        m_is_mkn = mkn_counts;
        if (!mkn_counts)
            initialise_kneser_ney(col, cst, max_node_depth);
        else
            initialise_modified_kneser_ney(col, cst, max_node_depth);
    }

    template <class t_cst, class t_node_type>
    uint32_t compute_contexts(t_cst& cst, t_node_type node, uint64_t& num_syms)
    {
        static std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
            cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
            cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        num_syms = 0;
        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms,
            preceding_syms, left, right);
        auto total_contexts = 0;
        auto node_depth = cst.depth(node);
        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb) {
                total_contexts++;
            }
            else {
                auto new_node = cst.node(new_lb, new_rb);
                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                }
                else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                }
            }
        }
        return total_contexts;
    }

    template <class t_cst, class t_node_type>
    uint32_t compute_contexts_mkn(t_cst& cst, t_node_type node,
        uint64_t& num_syms, uint64_t& f1prime,
        uint64_t& f2prime)
    {
        f1prime = 0;
        f2prime = 0;
        uint64_t all = 0;
        auto child = cst.select_child(node, 1);
        while (child != cst.root()) {
            auto lb = cst.lb(child);
            auto rb = cst.rb(child);

            static std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
                cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(
                cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
                cst.csa.sigma);
            typename t_cst::csa_type::size_type num_syms = 0;
            sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms,
                preceding_syms, left, right);
            if (num_syms == 1)
                f1prime++;
            if (num_syms == 2)
                f2prime++;
            all++;
            child = cst.sibling(child);
        }

        /* computes fb */
        auto total_contexts = 0;
        auto node_depth = cst.depth(node);

        static std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
            cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
            cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        num_syms = 0;
        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms,
            preceding_syms, left, right);

        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb) {
                total_contexts++;
            }
            else {
                auto new_node = cst.node(new_lb, new_rb);
                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                }
                else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                }
            }
        }
        return total_contexts;
    }

    template <class t_cst>
    void initialise_kneser_ney(collection& col, t_cst& cst,
        uint64_t max_node_depth)
    {
        sdsl::bit_vector tmp_bv(cst.nodes());
        sdsl::util::set_to_value(tmp_bv, 0);
        auto tmp_buffer_counts = sdsl::int_vector_buffer<32>(col.temp_file("counts"), std::ios::out);
        uint64_t num_syms = 0;

        auto root = cst.root();
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);

            while (itr != end) {
                auto node = *itr;
                if (itr.visit() == 2) {
                    auto node_id = cst.id(node);

                    auto str_depth = cst.depth(node);
                    if (str_depth <= max_node_depth) {
                        tmp_bv[node_id] = 1;
                        auto c = compute_contexts(cst, node, num_syms);
                        tmp_buffer_counts.push_back(c); // anchor
                        tmp_buffer_counts.push_back(0);
                        tmp_buffer_counts.push_back(0);
                        tmp_buffer_counts.push_back(num_syms);
                        tmp_buffer_counts.push_back(0);
                        tmp_buffer_counts.push_back(0);
                    }
                }
                else {
                    /* first visit */
                    if (!cst.is_leaf(node)) {
                        auto depth = cst.depth(node);
                        if (depth > max_node_depth) {
                            itr.skip_subtree();
                        }
                    }
                }
                ++itr;
            }
        }
        m_counts = vector_type(tmp_buffer_counts);
        tmp_buffer_counts.close(true);

        m_bv = bv_type(tmp_bv);
        tmp_bv.resize(0);
        m_bv_rank = rank_type(&m_bv);

        LOG(INFO) << "precomputed " << m_bv_rank(m_bv.size()) << " entries out of "
                  << m_bv.size() << " nodes";
    }

    // specific MKN implementation, 2-pass
    template <class t_cst>
    void initialise_modified_kneser_ney(collection& col, t_cst& cst,
        uint64_t max_node_depth)
    {
        sdsl::bit_vector tmp_bv(cst.nodes());
        sdsl::util::set_to_value(tmp_bv, 0);

        auto tmp_buffer_counts = sdsl::int_vector_buffer<32>(col.temp_file("counts"), std::ios::out);

        uint64_t num_syms = 0;
        uint64_t f1prime = 0, f2prime = 0;
        auto root = cst.root();
        std::vector<std::pair<uint64_t, uint64_t> > child_hist(max_node_depth + 2);
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);

            for (auto& v : child_hist)
                v = { 0, 0 };
            // std::map<uint64_t, std::pair<uint64_t, uint64_t> > child_hist;
            uint64_t node_depth = 1;
            auto prev = root;
            while (itr != end) {
                auto node = *itr;
                // auto node_depth = cst.node_depth(node);
                if (cst.parent(node) == prev)
                    node_depth++;

                if (itr.visit() == 2) { // anchor
                    node_depth--;
                    auto str_depth = cst.depth(node);
                    if (str_depth <= max_node_depth) {
                        auto node_id = cst.id(node);
                        tmp_bv[node_id] = 1;
                        auto& f12 = child_hist[node_depth];

                        // assert(cst.degree(node) >= f12.first + f12.second);

                        auto c = compute_contexts_mkn(cst, node, num_syms, f1prime, f2prime);

                        //
                        tmp_buffer_counts.push_back(c);
                        tmp_buffer_counts.push_back(f1prime);
                        tmp_buffer_counts.push_back(f2prime);
                        tmp_buffer_counts.push_back(num_syms);
                        tmp_buffer_counts.push_back(f12.first);
                        tmp_buffer_counts.push_back(f12.second);
                    }
                    child_hist[node_depth] = { 0, 0 };
                    // child_hist.erase(node_id);
                }
                else {
                    /* first visit */
                    if (!cst.is_leaf(node)) {
                        if (node_depth > max_node_depth) {
                            itr.skip_subtree();
                        }
                    }
                    int count = cst.size(node);
                    // auto parent_id = cst.id(cst.parent(node));
                    if (count == 1)
                        child_hist[node_depth - 1].first += 1;
                    else if (count == 2)
                        child_hist[node_depth - 1].second += 1;
                }
                prev = node;
                ++itr;
                // last_node_depth = depth;
            }
        }
        // store into compressed in-memory data structures
        m_bv = bv_type(tmp_bv);
        tmp_bv.resize(0);
        m_bv_rank = rank_type(&m_bv);

        m_counts = vector_type(tmp_buffer_counts);
        tmp_buffer_counts.close(true);

        LOG(INFO) << "precomputed " << m_bv_rank(m_bv.size()) << " entries out of "
                  << m_bv.size() << " nodes";
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
        std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += sdsl::serialize(m_bv, out, child, "bv");
        written_bytes += sdsl::serialize(m_bv_rank, out, child, "bv_rank");

        written_bytes += sdsl::serialize(m_counts, out, child, "counts");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        sdsl::load(m_bv, in);
        sdsl::load(m_bv_rank, in);
        m_bv_rank.set_vector(&m_bv);

        sdsl::load(m_counts, in);
        m_is_mkn = ((m_counts.size() / shift) > 0);
    }

    // FIXME: could do this a bit more efficiently, without decompressing m_bv
    // e.g., if its depth <= max_node_depth (but beware querying this for leaves)
    template <class t_cst, class t_node_type>
    bool is_precomputed(t_cst& cst, t_node_type node) const
    {
        auto id = cst.id(node);
        return m_bv[id];
    }

    template <class t_cst, class t_node_type>
    void lookup_f12(t_cst& cst, t_node_type node, uint64_t& f1,
        uint64_t& f2) const
    {
        auto timer = lm_bench::bench(timer_type::lookup_f12);
        assert(m_is_mkn);
        auto id = cst.id(node);
        // LOG(INFO) << "lookup_f12(" << id << ")";
        auto rank_in_vec = m_bv_rank(id);
        f1 = m_counts[(rank_in_vec * shift) + _f1]; 
        f2 = m_counts[(rank_in_vec * shift) + _f2];
    }

    template <class t_cst, class t_node_type>
    uint64_t lookup_fb(t_cst& cst, t_node_type node) const
    {
        auto timer = lm_bench::bench(timer_type::lookup_fb);
        auto id = cst.id(node);
        // LOG(INFO) << "lookup_fb(" << id << ")";
        auto rank_in_vec = m_bv_rank(id);
        return m_counts[(rank_in_vec * shift) + _fb];
    }

    template <class t_cst, class t_node_type>
    void lookup_f12prime(t_cst& cst, t_node_type node, uint64_t& f1prime,
        uint64_t& f2prime) const
    {
        auto timer = lm_bench::bench(timer_type::lookup_f12prime);
        assert(m_is_mkn);
        auto id = cst.id(node);
        // LOG(INFO) << "lookup_f12prime(" << id << ")";
        auto rank_in_vec = m_bv_rank(id);
        f1prime = m_counts[(rank_in_vec * shift) + _f1prime];
        f2prime = m_counts[(rank_in_vec * shift) + _f2prime];
    }

    template <class t_cst, class t_node_type>
    uint64_t lookup_b(t_cst& cst, t_node_type node) const
    {
        auto timer = lm_bench::bench(timer_type::lookup_b);
        auto id = cst.id(node);
        // LOG(INFO) << "lookup_b(" << id << ")";
        auto rank_in_vec = m_bv_rank(id);
        return m_counts[(rank_in_vec * shift) + _b];
    }
};
}
