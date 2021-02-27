# Detailed Example

The following code snippet demonstrates, how to create basic cubool based application
for sparse matrix-matrix multiplication and matrix-matrix element-wise addition
with cubool C API usage.

```c++
/************************************************/
/* Evaluate transitive closure for some graph G */
/************************************************/

/* Actual cubool C API */
#include <cubool/cubool.h>
#include <stdio.h>

/* Macro to check result of the function call */
#define CHECK(f) { cuBool_Status s = f; if (s != CUBOOL_STATUS_SUCCESS) return s; }

int main() {
    cuBool_Matrix A;
    cuBool_Matrix TC;

    /* System may not provide Cuda compatible device */
    CHECK(cuBool_Initialize(CUBOOL_HINT_NO));

    /* Input graph G */

    /*  -> (1) ->           */
    /*  |       |           */
    /* (0) --> (2) <--> (3) */

    /* Adjacency matrix in sparse format  */
    cuBool_Index n = 4;
    cuBool_Index e = 5;
    cuBool_Index rows[] = { 0, 0, 1, 2, 3 };
    cuBool_Index cols[] = { 1, 2, 2, 3, 2 };

    /* Create matrix */
    CHECK(cuBool_Matrix_New(&A, n, n));

    /* Fill the data */
    CHECK(cuBool_Matrix_Build(A, rows, cols, e, CUBOOL_HINT_VALUES_SORTED));

    /* Now we have created the following matrix */

    /*    [0][1][2][3]
    /* [0] .  1  1  .  */
    /* [1] .  .  1  .  */
    /* [2] .  .  .  1  */
    /* [3] .  .  1  .  */

    /* Create result matrix from source as copy */
    CHECK(cuBool_Matrix_Duplicate(A, &TC));

    /* Query current number on non-zero elements */
    cuBool_Index total = 0;
    cuBool_Index current;
    CHECK(cuBool_Matrix_Nvals(TC, &current));

    /* Loop while values are added */
    while (current != total) {
        total = current;

        /** Transitive closure step */
        CHECK(cuBool_MxM(TC, TC, TC, CUBOOL_HINT_ACCUMULATE));
        CHECK(cuBool_Matrix_Nvals(TC, &current));
    }

    /** Get result */
    cuBool_Index tc_rows[16], tc_cols[16];
    CHECK(cuBool_Matrix_ExtractPairs(TC, tc_rows, tc_cols, &total));

    /** Now tc_rows and tc_cols contain (i,j) pairs of the result G_tc graph */

    /*    [0][1][2][3]
    /* [0] .  1  1  1  */
    /* [1] .  .  1  1  */
    /* [2] .  .  1  1  */
    /* [3] .  .  1  1  */

    /* Output result size */
    printf("Nnz(tc)=%lli\n", (unsigned long long) total);

    for (cuBool_Index i = 0; i < total; i++)
        printf("(%u,%u) ", tc_rows[i], tc_cols[i]);

    /* Release resources */
    CHECK(cuBool_Matrix_Free(A));
    CHECK(cuBool_Matrix_Free(TC));

    /* Release library */
    return cuBool_Finalize() != CUBOOL_STATUS_SUCCESS;
}
```

Export path to library cubool, compile the file and run with the following
command (assuming, that source code is placed into tc.cpp file):

```shell script
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:path/to/folder/with/libcubool/"
$ gcc tc.cpp -o tc -I/path/to/cubool/include/dir/ -L/path/to/folder/with/libcubool/ -lcubool
$ ./tc 
```

The program will print the following output:

```
Nnz(tc)=9
(0,1) (0,2) (0,3) (1,2) (1,3) (2,2) (2,3) (3,2) (3,3)
```