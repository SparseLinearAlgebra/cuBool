#pragma once
#include <masked_matrix.h>
#include <detail/meta.h>
#include <detail/masked_mult.h>

namespace nsparse {

template <typename value_type, typename index_type, value_type zero, typename Mul, typename Add>
struct masked_spgemm_functor_t {
  masked_spgemm_functor_t(Mul mul, Add add) : masked_mult_functor(mul, add) {
  }

  /*
   * c += a * b
   */
  void operator()(masked_matrix<value_type, index_type>& c,
                  const masked_matrix<value_type, index_type>& a,
                  const masked_matrix<value_type, index_type>& b) {
    assert(&c != &a);
    assert(&c != &b);

    using namespace meta;

    constexpr auto config_mul = make_bin_seq<
        bin_info_t<mul_conf_t<128, 1024, 64>, 32768, 65536>,
        bin_info_t<mul_conf_t<128, 1024, 32>, 16384, 32768>,
        bin_info_t<mul_conf_t<128, 512, 32>, 8192, 16384>,
        bin_info_t<mul_conf_t<128, 1024, 8>, 4096, 8192>,
        bin_info_t<mul_conf_t<64, 1024, 4>, 1024, 4096>,
        bin_info_t<mul_conf_t<32, 512, 2>, 512, 1024>,
        bin_info_t<mul_conf_t<32, 256, 2>, 256, 512>,
        bin_info_t<mul_conf_t<32, 64, 4>, 64, 256>,
        bin_info_t<mul_conf_t<32, 32, 2>, 0, 64>
    >;

    masked_mult_functor(c.m_skeleton.m_rows, c.m_skeleton.m_col_index, c.m_skeleton.m_row_index,
                        c.m_values, a.m_skeleton.m_col_index, a.m_skeleton.m_row_index, a.m_values,
                        b.m_skeleton.m_col_index, b.m_skeleton.m_row_index, b.m_values, config_mul);
  }

 private:
  masked_mult_functor_t<value_type, index_type, zero, Mul, Add> masked_mult_functor;
};

}  // namespace nsparse