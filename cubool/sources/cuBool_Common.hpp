/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
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
/**********************************************************************************/

#ifndef CUBOOL_CUBOOL_COMMON_HPP
#define CUBOOL_CUBOOL_COMMON_HPP

#include <cubool/cubool.h>
#include <core/config.hpp>
#include <core/version.hpp>
#include <core/error.hpp>
#include <core/library.hpp>
#include <core/matrix.hpp>
#include <core/vector.hpp>
#include <cstring>

// State validation
#define CUBOOL_VALIDATE_LIBRARY                                                         \
    cubool::Library::validate();

// Arguments validation
#define CUBOOL_ARG_NOT_NULL(arg)                                                        \
    CHECK_RAISE_ERROR(arg != nullptr, InvalidArgument, "Passed null argument")

#define CUBOOL_BEGIN_BODY                                                               \
    try {

#define CUBOOL_END_BODY }                                                               \
    catch (const cubool::Error& err) {                                                  \
         cubool::Library::handleError(err);                                             \
         return err.getStatus();                                                        \
    }                                                                                   \
    catch (const std::exception& exc) {                                                 \
         cubool::Library::handleError(exc);                                             \
         return CUBOOL_STATUS_ERROR;                                                    \
    }                                                                                   \
    return cuBool_Status::CUBOOL_STATUS_SUCCESS;

#endif //CUBOOL_CUBOOL_COMMON_HPP
