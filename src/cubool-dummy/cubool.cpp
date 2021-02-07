/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/*                                                                                */
/**********************************************************************************/

#include <cubool/cubool.h>
#include <cubool-dummy/version.hpp>
#include <cubool-dummy/instance.hpp>
#include <exception>
#include <string>

#define CUBOOL_CHECK_INSTANCE(instance)                                                                 \
    if (!instance) {                                                                                    \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_CHECK_ARG_NOT_NULL(arg)                                                                  \
    if (!arg) {                                                                                         \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_BEGIN_BODY \
        try {

#define CUBOOL_END_BODY } \
        catch (const std::exception& err) { \
             return CuBoolStatus::CUBOOL_STATUS_ERROR; \
        }                      \
        return CuBoolStatus::CUBOOL_STATUS_SUCCESS; \

const char* CuBool_About_Get() {
    static const char about[] =
            "CuBool is a linear boolean algebra library primitives and operations for \n"
            "work with dense and sparse matrices written on the NVIDIA CUDA platform. The primary \n"
            "goal of the library is implementation, testing and profiling algorithms for\n"
            "solving *formal-language-constrained problems*, such as *context-free* \n"
            "and *recursive* path queries with various semantics for graph databases."
            ""
            "Contributors:"
            "- Semyon Grigorev (Github: https://github.com/gsvgit)\n"
            "- Egor Orachyov (Github: https://github.com/EgorOrachyov)";

    return about;
}

const char* CuBool_LicenseInfo_Get() {
    static const char license[] =
            "MIT License\n"
            "\n"
            "Copyright (c) 2020 JetBrains-Research\n"
            "\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            "copies of the Software, and to permit persons to whom the Software is\n"
            "furnished to do so, subject to the following conditions:\n"
            "\n"
            "The above copyright notice and this permission notice shall be included in all\n"
            "copies or substantial portions of the Software.\n"
            "\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
            "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
            "SOFTWARE.";

    return license;
}

CuBoolStatus CuBool_Version_Get(int* major, int* minor, int* version) {
    if (major) {
        *major = CUBOOL_VERSION_MAJOR;
    }

    if (minor) {
        *minor = CUBOOL_VERSION_MINOR;
    }

    if (version) {
        *version = CUBOOL_VERSION;
    }

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_DeviceCaps_Get(CuBoolDeviceCaps* deviceCaps) {
    if (!deviceCaps) {
        // Null caps structure
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_Instance_New(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance) {
    if (!instanceDesc) {
        // Instance descriptor could not be null
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!instance) {
        // Null to safe instance reference
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    auto instanceImpl = new cubool_dummy::Instance();
    *instance = (CuBoolInstance) instanceImpl;

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_Instance_Free(CuBoolInstance instance) {
    CUBOOL_CHECK_INSTANCE(instance);

    auto instanceImpl = (cubool_dummy::Instance*) instance;
    delete instanceImpl;

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_HostDevice_Sync(CuBoolInstance instance) {
    CUBOOL_CHECK_INSTANCE(instance);
    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_MatrixDense_New(CuBoolInstance instance, CuBoolMatrixDense *matrix, CuBoolIndex_t nrows, CuBoolIndex_t ncols) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        cubool_dummy::Matrix* matrixImpl = instanceImpl->createMatrix(nrows, ncols);
        *matrix = (CuBoolMatrixDense) matrixImpl;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Free(CuBoolInstance instance, CuBoolMatrixDense matrix) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->destroyMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Resize(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolIndex_t nrows, CuBoolIndex_t ncols) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->resize(nrows, ncols);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Build(CuBoolInstance instance, CuBoolMatrixDense matrix, const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_ExtractPairsExt(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);
    CUBOOL_CHECK_ARG_NOT_NULL(nvals);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        *rows = nullptr;
        *cols = nullptr;
        *nvals = 0;
    CUBOOL_END_BODY
}


CuBoolStatus CuBool_Vals_Free(CuBoolInstance instance, CuBoolIndex_t* vals) {
    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(vals);

    CUBOOL_BEGIN_BODY
        // vals == null
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_MxM(CuBoolInstance instance, CuBoolMatrixDense r, CuBoolMatrixDense a, CuBoolMatrixDense b) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto rImpl = (cubool_dummy::Matrix*) r;
    auto aImpl = (cubool_dummy::Matrix*) a;
    auto bImpl = (cubool_dummy::Matrix*) b;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(rImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_MultSum(CuBoolInstance instance, CuBoolMatrixDense r, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto rImpl = (cubool_dummy::Matrix*) r;
    auto aImpl = (cubool_dummy::Matrix*) a;
    auto bImpl = (cubool_dummy::Matrix*) b;
    auto cImpl = (cubool_dummy::Matrix*) c;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);
    CUBOOL_CHECK_ARG_NOT_NULL(c);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(cImpl);
        instanceImpl->validateMatrix(rImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_New(CuBoolInstance instance, CuBoolMatrix *matrix, CuBoolIndex_t nrows, CuBoolIndex_t ncols) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        cubool_dummy::Matrix* matrixImpl = instanceImpl->createMatrix(nrows, ncols);
        *matrix = (CuBoolMatrix) matrixImpl;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Resize(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolIndex_t nrows, CuBoolIndex_t ncols) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->resize(nrows, ncols);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Build(CuBoolInstance instance, CuBoolMatrix matrix, const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_ExtractPairsExt(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);
    CUBOOL_CHECK_ARG_NOT_NULL(nvals);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        *rows = nullptr;
        *cols = nullptr;
        *nvals = 0;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Duplicate(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolMatrix *duplicated) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto matrixImpl = (cubool_dummy::Matrix *) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(duplicated);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        cubool_dummy::Matrix* matrixDuplicated = instanceImpl->createMatrix(matrixImpl->getNumRows(), matrixImpl->getNumCols());
        *duplicated = (CuBoolMatrix) matrixDuplicated;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Transpose(CuBoolInstance instance, CuBoolMatrix result, CuBoolMatrix matrix) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto resultImpl = (cubool_dummy::Matrix *) result;
    auto matrixImpl = (cubool_dummy::Matrix *) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(result);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        instanceImpl->validateMatrix(resultImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Nvals(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolSize_t *nvals) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto matrixImpl = (cubool_dummy::Matrix *) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(nvals);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        *nvals = 0;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Nrows(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolIndex_t *nrows) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto matrixImpl = (cubool_dummy::Matrix *) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(nrows);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        *nrows = matrixImpl->getNumRows();
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Ncols(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolIndex_t *ncols) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto matrixImpl = (cubool_dummy::Matrix *) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(ncols);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        *ncols = matrixImpl->getNumCols();
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Free(CuBoolInstance instance, CuBoolMatrix matrix) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto matrixImpl = (cubool_dummy::Matrix*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->destroyMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_EWise_Add(CuBoolInstance instance, CuBoolMatrix r, CuBoolMatrix a) {
    auto instanceImpl = (cubool_dummy::Instance *) instance;
    auto rImpl = (cubool_dummy::Matrix *) r;
    auto aImpl = (cubool_dummy::Matrix *) a;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(rImpl);
        instanceImpl->validateMatrix(aImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MxM(CuBoolInstance instance, CuBoolMatrix r, CuBoolMatrix a, CuBoolMatrix b) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto rImpl = (cubool_dummy::Matrix*) r;
    auto aImpl = (cubool_dummy::Matrix*) a;
    auto bImpl = (cubool_dummy::Matrix*) b;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(rImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Kron(CuBoolInstance instance, CuBoolMatrix r, CuBoolMatrix a, CuBoolMatrix b) {
    auto instanceImpl = (cubool_dummy::Instance*) instance;
    auto rImpl = (cubool_dummy::Matrix*) r;
    auto aImpl = (cubool_dummy::Matrix*) a;
    auto bImpl = (cubool_dummy::Matrix*) b;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(rImpl);
    CUBOOL_END_BODY
}