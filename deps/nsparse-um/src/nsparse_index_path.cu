#include "nsparse.h"
#include "masked_spgemm.h"

#include <vector>
#include <unified_allocator.h>

using index_type = uint32_t;
using value_type = uint64_t;

std::vector<nsparse::masked_matrix<value_type, index_type>> index_path(
    std::vector<nsparse::matrix<bool, index_type>> init_matrices,
    std::vector<nsparse::matrix<bool, index_type>> final_matrices,
    const std::vector<std::tuple<int, int, int>>& evaluation_plan, index_type graph_size) {
  assert(init_matrices.size() == final_matrices.size());
  auto nonterm_count = init_matrices.size();
  std::vector<nsparse::masked_matrix<value_type, index_type>> masked_matrices;
  masked_matrices.reserve(nonterm_count);

  constexpr value_type zero = std::numeric_limits<value_type>::max();

  {
    value_type edge = 1;
    edge <<= sizeof(value_type) * 8 / 2;
    edge += std::numeric_limits<index_type>::max();

    auto identity = nsparse::masked_matrix<value_type, index_type>::identity(graph_size, 0);
    auto id_mul = [] __device__(value_type lhs, value_type rhs, index_type a_col) -> value_type {
      return lhs;
    };
    auto id_add = [] __device__(value_type * lhs, value_type rhs) -> void { *lhs = rhs; };

    nsparse::masked_spgemm_functor_t<value_type, index_type, zero, decltype(id_mul),
                                     decltype(id_add)>
        masked_id_spgemm(id_mul, id_add);

    for (auto i = 0; i < nonterm_count; i++) {
      masked_matrices.emplace_back(std::move(final_matrices[i]), -1);

      index_type left_size = init_matrices[i].m_vals;

      nsparse::masked_matrix<value_type, index_type> left(
          std::move(init_matrices[i]), nsparse::managed_vector<value_type>(left_size, edge));

      masked_id_spgemm(masked_matrices.back(), left, identity);
      cudaDeviceSynchronize();
    }
  }

  {
    auto mul = [] __device__(value_type lhs, value_type rhs, index_type a_col) -> value_type {
      value_type mult_res = max(lhs, rhs);
      mult_res >>= sizeof(value_type) * 8 / 2;
      mult_res++;
      mult_res <<= sizeof(value_type) * 8 / 2;
      mult_res += a_col;
      return mult_res;
    };

    auto add = [] __device__(value_type * lhs, value_type rhs) -> void {
      static_assert(sizeof(unsigned long long) == sizeof(value_type));
      //      atomicMin((unsigned long long*)lhs, (unsigned long long)rhs);
      //      *lhs = min(*lhs, rhs);
      unsigned long long int old = (unsigned long long int)(*lhs);
      unsigned long long int expected;

      do {
        expected = old;
        old = atomicCAS((unsigned long long int*)lhs, expected,
                        min((unsigned long long int)rhs, expected));
      } while (expected != old);
    };

    nsparse::masked_spgemm_functor_t<value_type, index_type, zero, decltype(mul), decltype(add)>
        masked_spgemm(mul, add);

    for (auto& item : evaluation_plan) {
      masked_spgemm(masked_matrices[std::get<0>(item)], masked_matrices[std::get<1>(item)],
                    masked_matrices[std::get<2>(item)]);
    }
    cudaDeviceSynchronize();
  }

  return masked_matrices;
}
