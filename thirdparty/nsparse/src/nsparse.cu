#include "matrix.h"
#include "nsparse.h"
#include "masked_matrix.h"

#include <chrono>
#include <grammar.h>
#include <item_mapper.h>
#include <response.h>
#include <vector>

using namespace std::chrono;
using index_type = uint32_t;
using value_type = uint64_t;

std::vector<nsparse::matrix<bool, index_type>> nsparse_prepare(const Grammar* grammar,
                                                               const GrB_Matrix* relations,
                                                               const char** relations_names,
                                                               size_t relations_count,
                                                               size_t graph_size);

std::pair<int, std::vector<std::tuple<int, int, int>>> nsparse_binary(
    const Grammar* grammar, std::vector<nsparse::matrix<bool, index_type>>& matrices);

int nsparse_cfpq(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                 const char** relations_names, size_t relations_count, size_t graph_size) {
  auto t1 = high_resolution_clock::now();

  std::vector<nsparse::matrix<bool, index_type>> matrices =
      nsparse_prepare(grammar, relations, relations_names, relations_count, graph_size);
  cudaDeviceSynchronize();

  auto t2 = high_resolution_clock::now();
  response->time_to_prepare += duration<double, seconds::period>(t2 - t1).count();

  auto relational_sem_info = nsparse_binary(grammar, matrices);
  cudaDeviceSynchronize();

  response->iteration_count = relational_sem_info.first;

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char* nonterm;

    nvals = matrices[i].m_vals;
    nonterm = ItemMapper_Map((ItemMapper*)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}

std::vector<nsparse::masked_matrix<value_type, index_type>> index_path(
    std::vector<nsparse::matrix<bool, index_type>> init_matrices,
    std::vector<nsparse::matrix<bool, index_type>> final_matrices,
    const std::vector<std::tuple<int, int, int>>& evaluation_plan, index_type graph_size,
    index_type nonterm_count);

int nsparse_cfpq_index(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                       const char** relations_names, size_t relations_count, size_t graph_size) {
  auto t1 = high_resolution_clock::now();

  std::vector<nsparse::matrix<bool, index_type>> matrices =
      nsparse_prepare(grammar, relations, relations_names, relations_count, graph_size);
  std::vector<nsparse::matrix<bool, index_type>> matrices_copied(matrices);
  cudaDeviceSynchronize();

  auto t2 = high_resolution_clock::now();
  response->time_to_prepare += duration<double, seconds::period>(t2 - t1).count();

  auto relational_sem_info = nsparse_binary(grammar, matrices);

  auto indexed_paths = index_path(std::move(matrices_copied), std::move(matrices), relational_sem_info.second,
                                  graph_size, grammar->nontermMapper.count);

  cudaDeviceSynchronize();

  response->iteration_count = relational_sem_info.first;

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char* nonterm;

    nvals = indexed_paths[i].m_skeleton.m_vals;
    nonterm = ItemMapper_Map((ItemMapper*)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}