#include "matrix.h"

#include <grammar.h>
#include <item_mapper.h>
#include <response.h>
#include <vector>

using index_type = uint32_t;

std::vector<nsparse::matrix<bool, index_type>> nsparse_prepare(
    const Grammar* grammar, const GrB_Matrix* relations,
    const char** relations_names, size_t relations_count, size_t graph_size) {
  size_t nonterm_count = grammar->nontermMapper.count;

  std::vector<nsparse::matrix<bool, index_type>> matrices(
      nonterm_count, {static_cast<index_type>(graph_size), static_cast<index_type>(graph_size)});

  // Initialize matrices
  for (int i = 0; i < relations_count; i++) {
    const char* terminal = relations_names[i];

    MapperIndex terminal_id =
        ItemMapper_GetPlaceIndex((ItemMapper*)&grammar->tokenMapper, terminal);
    if (terminal_id != grammar->tokenMapper.count) {
      for (int j = 0; j < grammar->simple_rules_count; j++) {
        const SimpleRule* simpleRule = &grammar->simple_rules[j];
        if (simpleRule->r == terminal_id) {
          GrB_Matrix tmp_matrix;
          GrB_Matrix_dup(&tmp_matrix, relations[i]);

          GrB_Type tp;
          GrB_Index nrows, ncols, nvals;
          int64_t nonempty;

          GrB_Index* col_idx;
          GrB_Index* row_idx;
          void* vals;

          GrB_Descriptor desc;
          GrB_Descriptor_new(&desc);

          GxB_Matrix_export_CSR(&tmp_matrix, &tp, &nrows, &ncols, &nvals, &nonempty, &row_idx,
                                &col_idx, &vals, desc);

          thrust::device_vector<index_type> col_index(col_idx, col_idx + nvals);
          thrust::device_vector<index_type> row_index(row_idx, row_idx + nrows + 1);

          matrices[simpleRule->l] = {
              std::move(col_index), std::move(row_index), static_cast<index_type>(graph_size),
              static_cast<index_type>(graph_size), static_cast<index_type>(nvals)};

//          delete[] col_idx;
//          delete[] row_idx;
//          delete[](bool*) vals;

//          GrB_Descriptor_free(&desc);
        }
      }
    }
  }

  return matrices;
}