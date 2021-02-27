#include "spgemm.h"
#include "nsparse.h"

#include <grammar.h>
#include <item_mapper.h>
#include <response.h>
#include <vector>

using index_type = uint32_t;

std::pair<int, std::vector<std::tuple<int, int, int>>> nsparse_binary(
    const Grammar* grammar, std::vector<nsparse::matrix<bool, index_type>>& matrices) {
  size_t nonterm_count = grammar->nontermMapper.count;

  std::vector<std::tuple<int, int, int>> evaluation_plan;
  std::vector<bool> changed(nonterm_count, true);

  nsparse::spgemm_functor_t<bool, index_type> spgemm_functor;

  int iteration_count = 0;

  while (std::find(changed.begin(), changed.end(), true) != changed.end()) {
    iteration_count++;

    std::vector<uint> sizes_before(nonterm_count);
    for (auto i = 0; i < nonterm_count; i++) {
      sizes_before[i] = matrices[i].m_vals;
    }

    for (int i = 0; i < grammar->complex_rules_count; ++i) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      if (!changed[nonterm2] && !changed[nonterm3])
        continue;

      if (matrices[nonterm2].m_vals == 0 || matrices[nonterm3].m_vals == 0)
        continue;

      auto vals_before = matrices[nonterm1].m_vals;
      auto new_mat = spgemm_functor(matrices[nonterm1], matrices[nonterm2], matrices[nonterm3]);
      auto vals_after = new_mat.m_vals;

      if (vals_after != vals_before) {
        changed[nonterm1] = true;
        evaluation_plan.emplace_back(nonterm1, nonterm2, nonterm3);
      }

      matrices[nonterm1] = std::move(new_mat);
    }

    for (auto i = 0; i < nonterm_count; i++) {
      changed[i] = sizes_before[i] != matrices[i].m_vals;
    }
  }

  return {iteration_count, evaluation_plan};
}