/**********************************************************************************/
/*                                                                                */
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
/*                                                                                */
/**********************************************************************************/

#ifndef CUBOOL_CUBOOL_COMMON_HPP
#define CUBOOL_CUBOOL_COMMON_HPP

#include <cubool/cubool.h>
#include <core/config.hpp>
#include <core/version.hpp>

#define CUBOOL_CHECK_INSTANCE_LATE_FREE(instance)                                                       \
    if ((cubool::Instance*) instance != cubool::Instance::getInstancePtr()) {                           \
        return cubool::Instance::isManagedUsageAllowed() ?                                              \
            CUBOOL_STATUS_SUCCESS : CUBOOL_STATUS_INVALID_ARGUMENT;                                     \
    }

#define CUBOOL_CHECK_INSTANCE(instance)                                                                 \
    if ((cubool::Instance*) instance != cubool::Instance::getInstancePtr()) {                           \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_CHECK_ARG_NOT_NULL(arg)                                                                  \
    if (!arg) {                                                                                         \
        std::string message = "Passed null ptr to arg: ";                                               \
        message += #arg ;                                                                               \
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, message.c_str());                     \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_BEGIN_BODY                                                                               \
    try {

#define CUBOOL_END_BODY }                                                                               \
    catch (const cubool::details::Error& err) {                                                         \
         instanceImpl->sendMessage(err.status(), err.what());                                           \
         return err.status();                                                                           \
    }                                                                                                   \
    return CuBoolStatus::CUBOOL_STATUS_SUCCESS;

#endif //CUBOOL_CUBOOL_COMMON_HPP
