cmake_minimum_required(VERSION 3.15)
project(nsparse_um LANGUAGES CXX CUDA)

add_library(nsparse_um INTERFACE)
target_include_directories(nsparse_um INTERFACE include/)
target_link_libraries(nsparse_um INTERFACE cub)
target_compile_options(nsparse_um INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --expt-extended-lambda>)

if (CUBOOL_BUILD_NSPARSE_TESTS)
    add_subdirectory(test)
endif()
