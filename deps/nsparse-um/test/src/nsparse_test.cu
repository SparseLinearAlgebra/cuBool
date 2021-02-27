#include <gtest/gtest.h>
#include <nsparse/detail/masked_mult.cuh>
#include <nsparse/masked_spgemm.h>
#include <nsparse/spgemm.h>
#include <random>

using index_type = unsigned int;
using alloc_type = nsparse::managed<index_type>;

using b_mat = std::vector<std::vector<bool>>;

std::pair<std::vector<index_type>, std::vector<index_type>> dense_to_csr(const b_mat& matrix) {
  std::vector<index_type> col_index;
  std::vector<index_type> row_index;

  index_type size = 0;
  for (const auto& row : matrix) {
    row_index.push_back(size);
    for (unsigned int i = 0; i < row.size(); i++) {
      if (row[i]) {
        col_index.push_back(i);
        size++;
      }
    }
  }
  row_index.push_back(size);

  return {col_index, row_index};
}

b_mat matrix_generator(size_t rows, size_t cols, float density);

template <typename T>
std::vector<T> value_generator(size_t size, T min, T max) {
  std::mt19937 gen;
  std::uniform_int_distribution<T> urd(min, max);

  std::vector<T> values;
  for (auto i = 0; i < size; i++) {
    values.push_back(urd(gen));
  }

  return values;
}

b_mat mult(const b_mat& a, const b_mat& b);

b_mat sum(const b_mat& a, const b_mat& b);

template <typename T>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T> vec) {
  for (auto i = 0; i < vec.size(); i++) {
    os << vec[i] << " ";
  }
  return os;
}

nsparse::matrix<bool, index_type, alloc_type> dense_to_gpu_csr(const b_mat& matrix) {
  auto m = dense_to_csr(matrix);
  return nsparse::matrix<bool, index_type, alloc_type>(
          m.first, m.second, matrix.size(), matrix[0].size(), m.second.back());
}

class NsparseCountNonZeroTest : public testing::Test {
 protected:
  static void eval(const b_mat& c, const b_mat& a, const b_mat& b) {
    b_mat r = sum(c, mult(a, b));

    auto sprsA = dense_to_csr(a);
    auto sprsB = dense_to_csr(b);
    auto sprsC = dense_to_csr(c);
    auto sprsR = dense_to_csr(r);

    nsparse::matrix<bool, index_type, alloc_type>
        A(sprsA.first, sprsA.second, a.size(), a[0].size(), sprsA.second.back());

    nsparse::matrix<bool, index_type, alloc_type>
        B(sprsB.first, sprsB.second, b.size(), b[0].size(), sprsB.second.back());

    nsparse::matrix<bool, index_type, alloc_type>
        C(sprsC.first, sprsC.second, c.size(), c[0].size(), sprsC.second.back());

    nsparse::spgemm_functor_t<bool, index_type, alloc_type> spgemm_functor;
    auto res = spgemm_functor(C, A, B);

    ASSERT_EQ(sprsR.second, res.m_row_index);
    ASSERT_EQ(sprsR.first, res.m_col_index);
  }
};

template <typename value_type>
void test_masked(const b_mat& a, const b_mat& b) {
  const b_mat c = mult(a, b);

  auto sprsA = dense_to_csr(a);
  auto sprsB = dense_to_csr(b);
  auto sprsC = dense_to_csr(c);

  auto maxH = 1025;
  constexpr value_type zero = std::numeric_limits<value_type>::max();

  std::vector<value_type> a_values = value_generator<value_type>(sprsA.first.size(), 0, maxH);
  std::vector<value_type> b_values = value_generator<value_type>(sprsB.first.size(), 0, maxH);
  std::for_each(a_values.begin(), a_values.end(),
                [](value_type& elem) { elem <<= sizeof(value_type) * 8 / 2; });
  std::for_each(b_values.begin(), b_values.end(),
                [](value_type& elem) { elem <<= sizeof(value_type) * 8 / 2; });

  std::vector<value_type> expected_c_values(sprsC.first.size(), zero);
  {
    auto rows = a.size();
    for (auto row = 0; row < rows; row++) {
      auto c_row_begin = sprsC.second[row];
      auto c_row_end = sprsC.second[row + 1];

      auto a_row_begin = sprsA.second[row];
      auto a_row_end = sprsA.second[row + 1];

      for (auto i = a_row_begin; i < a_row_end; i++) {
        auto a_col = sprsA.first[i];
        auto a_value = a_values[i];

        if (a_value == zero)
          continue;

        auto b_row_begin = sprsB.second[a_col];
        auto b_row_end = sprsB.second[a_col + 1];

        for (auto j = b_row_begin; j < b_row_end; j++) {
          auto b_col = sprsB.first[j];
          auto b_value = b_values[j];

          if (b_value == zero)
            continue;

          value_type mult_res = std::max(a_value, b_value);
          mult_res >>= sizeof(value_type) * 8 / 2;
          mult_res++;
          mult_res <<= sizeof(value_type) * 8 / 2;
          mult_res += a_col;

          auto it =
              std::find(sprsC.first.begin() + c_row_begin, sprsC.first.begin() + c_row_end, b_col);
          assert(it != sprsC.first.end());
          auto pos = it - sprsC.first.begin();
          expected_c_values[pos] = std::min(expected_c_values[pos], mult_res);
        }
      }
    }
  }

  auto gpu_c = dense_to_gpu_csr(c);
  auto gpu_a = dense_to_gpu_csr(a);
  auto gpu_b = dense_to_gpu_csr(b);

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
    //    atomicMin((unsigned long long*)lhs, (unsigned long long)rhs);

    unsigned long long int old = (unsigned long long int)(*lhs);
    unsigned long long int expected;

    do {
      expected = old;
      old = atomicCAS((unsigned long long int*)lhs, expected,
                      min((unsigned long long int)rhs, expected));
    } while (expected != old);
  };

  nsparse::masked_matrix<value_type, index_type> masked_a(gpu_a, a_values);
  nsparse::masked_matrix<value_type, index_type> masked_b(gpu_b, b_values);
  nsparse::masked_matrix<value_type, index_type> masked_c(gpu_c, -1);

  nsparse::masked_spgemm_functor_t<value_type, index_type, zero, decltype(mul), decltype(add)> masked_spgemm(mul, add);

  masked_spgemm(masked_c, masked_a, masked_b);
  ASSERT_EQ(expected_c_values, masked_c.m_values);
}

TEST_F(NsparseCountNonZeroTest, multMaskedSmall) {
  size_t a = 10;
  size_t b = 15;
  size_t c = 20;

  for (float density = 0.01; density <= 1; density += 0.1) {
    test_masked<uint64_t>(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, multMaskedMedium) {
  size_t a = 500;
  size_t b = 600;
  size_t c = 700;

  for (float density = 0.01; density <= 0.5; density += 0.1) {
    test_masked<uint64_t>(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, multMaskedBig) {
  size_t a = 1000;
  size_t b = 1100;
  size_t c = 1200;

  for (float density = 0.01; density <= 0.2; density += 0.05) {
    test_masked<uint64_t>(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzSmall) {
  eval(
      {
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
      },
      {
          {0, 1, 0, 0, 1, 0},
          {1, 0, 1, 0, 1, 0},
          {0, 0, 0, 0, 0, 0},
          {0, 1, 1, 0, 0, 0},
      },
      {
          {0, 0, 1, 0, 0},
          {1, 0, 1, 0, 1},
          {1, 1, 1, 1, 1},
          {0, 0, 0, 0, 0},
          {0, 1, 0, 0, 0},
          {0, 0, 1, 1, 1},
      });
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedSmall) {
  size_t a = 100;
  size_t b = 150;
  size_t c = 200;

  for (float density = 0.01; density <= 1; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedMedium) {
  size_t a = 500;
  size_t b = 600;
  size_t c = 700;

  for (float density = 0.01; density <= 0.5; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedBig) {
  size_t a = 1000;
  size_t b = 1100;
  size_t c = 1200;

  for (float density = 0.01; density <= 0.2; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedGlobalHashTable) {
  size_t a = 100;
  size_t b = 500;
  size_t c = 5000;

  eval(matrix_generator(a, c, 0.5), matrix_generator(a, b, 0.5), matrix_generator(b, c, 0.5));
}