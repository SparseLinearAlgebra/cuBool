#pragma once

#define EXPAND_SIDE_EFFECTS(PATTERN)     \
  ::nsparse::meta::detail::expand_type { \
    0, ((PATTERN), 0)...                 \
  }

namespace nsparse {
namespace meta {

enum exec_policy_t { pwarp_row, block_row, global_row };

template <size_t block_sz, size_t cache_sz, size_t cache_stp>
struct mul_conf_t {
  static constexpr size_t block_size = block_sz;
  static constexpr size_t cache_size = cache_sz;
  static constexpr size_t cache_step = cache_stp;
};

template <size_t block_sz>
struct merge_conf_t {
  static constexpr size_t block_size = block_sz;
};

template <exec_policy_t policy, size_t block_sz>
struct nz_conf_t {
  static constexpr size_t block_size = block_sz;
  static constexpr exec_policy_t exec_policy = policy;
};

template <typename conf, size_t min_brd, size_t max_brd,
          size_t bin_idx = std::numeric_limits<size_t>::max()>
struct bin_info_t {
  static constexpr size_t min_border = min_brd;
  static constexpr size_t max_border = max_brd;
  static constexpr size_t bin_index = bin_idx;

  using config_t = conf;

  template <size_t idx>
  using with_bin_idx = bin_info_t<config_t, min_brd, max_brd, idx>;
};

namespace detail {

template <typename... Bins, size_t... Idx>
constexpr auto make_bin_seq_impl(std::index_sequence<Idx...>) {
  return std::tuple<typename Bins::with_bin_idx<Idx>...>{};
}

struct expand_type {
  template <typename... T>
  expand_type(T&&...) {
  }
};

template <exec_policy_t policy>
constexpr auto filter() {
  return std::tuple<>{};
}

template <exec_policy_t policy, typename Border, typename... Borders>
constexpr auto filter() {
  return std::conditional_t<Border::config_t::exec_policy == policy,
                            decltype(
                                std::tuple_cat(std::tuple<Border>{}, filter<policy, Borders...>())),
                            decltype(filter<policy, Borders...>())>{};
}

template <typename = void>
constexpr size_t max() {
  return 0;
}

template <size_t Value, size_t... Values>
constexpr size_t max() {
  return std::max(Value, max<Values...>());
}

constexpr bool all_of(std::initializer_list<bool> list) {
  for (auto item : list) {
    if (!item)
      return false;
  }
  return true;
}

}  // namespace detail

template <typename... Ts>
constexpr auto make_bin_seq =
    detail::make_bin_seq_impl<Ts...>(std::make_index_sequence<sizeof...(Ts)>{});

template <bool... values>
constexpr bool all_of = detail::all_of({values...});

template <exec_policy_t policy, typename... Borders>
constexpr auto filter = detail::filter<policy, Borders...>();

template <typename... Borders>
constexpr size_t max_bin = detail::max<Borders::bin_index...>();

template <typename... Borders>
__host__ __device__ size_t select_bin(size_t sz, size_t unused) {
  constexpr size_t min_borders[] = {Borders::min_border...};
  constexpr size_t max_borders[] = {Borders::max_border...};
  constexpr size_t bin_index[] = {Borders::bin_index...};

  for (size_t i = 0; i < sizeof...(Borders); i++) {
    if (sz > min_borders[i] && sz <= max_borders[i]) {
      return bin_index[i];
    }
  }

  return unused;
}

}  // namespace meta

}  // namespace nsparse