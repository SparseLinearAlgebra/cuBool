#include "nsparse.h"

#include <generic.hpp>

#include "spgemm.h"
#include "matrix.h"
#include "masked_matrix.h"
#include "unified_allocator.h"

#include <chrono>
#include <iostream>

using index_type = uint32_t;
using value_type = uint64_t;

namespace {
struct matrix {
  matrix() = default;
  explicit matrix(const GrB_Matrix& other) {
    GrB_Matrix tmp_matrix;
    GrB_Matrix_dup(&tmp_matrix, other);

    GrB_Type tp;
    GrB_Index nrows, ncols, nvals;
    int64_t nonempty;

    GrB_Index* col_idx;
    GrB_Index* row_idx;
    void* vals;

    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);

    GxB_Matrix_export_CSR(&tmp_matrix, &tp, &nrows, &ncols, &nvals, &nonempty, &row_idx, &col_idx,
                          &vals, desc);

    nsparse::managed_vector<GrB_Index> col_host(col_idx, col_idx + nvals);
    nsparse::managed_vector<GrB_Index> row_host(row_idx, row_idx + nrows + 1);

    nsparse::managed_vector<index_type> col_index(col_host);
    nsparse::managed_vector<index_type> row_index(row_host);
    data_ = nsparse::matrix<bool, index_type>{
        std::move(col_index), std::move(row_index), static_cast<index_type>(nrows),
        static_cast<index_type>(ncols), static_cast<index_type>(nvals)};
  }

  matrix(matrix&&) = default;
  matrix& operator=(matrix&&) = default;

  matrix(const matrix&) = default;
  matrix& operator=(const matrix&) = default;

  explicit matrix(size_t sz) : data_{static_cast<index_type>(sz), static_cast<index_type>(sz)} {
  }

  explicit matrix(nsparse::matrix<bool, index_type> data) : data_(std::move(data)) {
  }

  auto vals() const {
    return data_.m_vals;
  }

  nsparse::matrix<bool, index_type> data_;
};

struct functor {
  matrix operator()(const matrix& d, const matrix& a, const matrix& b) {
    auto res = f_(d.data_, a.data_, b.data_);
    return matrix{std::move(res)};
  }

  nsparse::spgemm_functor_t<bool, index_type> f_;
};

}  // namespace

int nsparse_cfpq(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                 const char** relations_names, size_t relations_count, size_t graph_size) {
  auto matrices = algorithms::matrix_init<matrix>(grammar, response, relations, relations_names,
                                                  relations_count, graph_size);
  auto res = algorithms::matrix_closure<matrix, functor>(grammar, matrices);
  algorithms::fill_response<matrix>(grammar, matrices, response, res.first);
  return 0;
}

std::vector<nsparse::masked_matrix<value_type, index_type>> index_path(
    std::vector<nsparse::matrix<bool, index_type>> init_matrices,
    std::vector<nsparse::matrix<bool, index_type>> final_matrices,
    const std::vector<std::tuple<int, int, int>>& evaluation_plan, index_type graph_size);

int nsparse_cfpq_index(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                       const char** relations_names, size_t relations_count, size_t graph_size) {
  auto init_matrices = algorithms::matrix_init<matrix>(
      grammar, response, relations, relations_names, relations_count, graph_size);
  auto final_matrices = init_matrices;

  auto t1 = std::chrono::high_resolution_clock::now();
  auto res = algorithms::matrix_closure<matrix, functor>(grammar, final_matrices);
  auto t2 = std::chrono::high_resolution_clock::now();
  response->time_to += std::chrono::duration<double, std::chrono::seconds::period>(t2 - t1).count();

  std::vector<nsparse::matrix<bool, index_type>> init_matrices_, final_matrices_;
  for (auto& item : init_matrices) {
    init_matrices_.emplace_back(std::move(item.data_));
  }

  for (auto& item : final_matrices) {
    final_matrices_.emplace_back(std::move(item.data_));
  }

  auto path_index_matrices =
      index_path(std::move(init_matrices_), std::move(final_matrices_), res.second, graph_size);
  algorithms::fill_response<>(grammar, path_index_matrices, response, res.first);

  return 0;
}
