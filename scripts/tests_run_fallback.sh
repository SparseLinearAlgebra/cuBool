# Runs all tests executables
# Invoke this script within build directory
./cubool/tests/test_library_api
./cubool/tests/test_matrix_misc
./cubool/tests/test_matrix_ewiseadd --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_extract_sub_matrix --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_kronecker --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_mxm --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_reduce --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_setup --gtest_filter=*.*Fallback
./cubool/tests/test_matrix_transpose --gtest_filter=*.*Fallback
