cmake_minimum_required(VERSION 3.11)
project(nsparse CXX CUDA)

add_library(nsparse INTERFACE)
target_include_directories(nsparse INTERFACE include/)
target_compile_options(nsparse INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --expt-extended-lambda -Xptxas -O3 >)

add_subdirectory(test)
