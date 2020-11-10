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

#ifndef CUBOOL_ERROR_HPP
#define CUBOOL_ERROR_HPP

#include <cubool/cubool.h>
#include <exception>
#include <string>

namespace cubool {
    namespace details {

        /** Base class for any exception thrown in the cubool backend implementation */
        class Error: public std::exception {
        public:
            Error(std::string &&message, CuBoolStatus status): std::exception(), mWhat(std::move(message)), mStatus(status) {}
            Error(const Error& error) noexcept = default;
            Error(Error&& error) noexcept = default;
            ~Error() noexcept override = default;

            /** @return Descriptive error message text */
            const char *what() const noexcept override { return mWhat.c_str(); }

            /** @return Status number which matches the C API error identification */
            CuBoolStatus status() const noexcept { return mStatus; }

        private:
            std::string mWhat;
            CuBoolStatus mStatus;
        };

        template <CuBoolStatus S>
        class TStatusError: public Error {
        public:
            explicit TStatusError(std::string&& message) noexcept: Error(std::move(message), S) {}
            TStatusError(const TStatusError<S>& error) noexcept = default;
            TStatusError(TStatusError<S>&& error) noexcept = default;
            ~TStatusError() noexcept override = default;
        };

        using DeviceError = TStatusError<CuBoolStatus::CUBOOL_STATUS_DEVICE_ERROR>;
        using InvalidArgument = TStatusError<CuBoolStatus::CUBOOL_STATUS_INVALID_ARGUMENT>;
        using InvalidState = TStatusError<CuBoolStatus::CUBOOL_STATUS_INVALID_STATE>;
        using MemOpFailed = TStatusError<CuBoolStatus::CUBOOL_STATUS_MEM_OP_FAILED>;

    }
}

#endif //CUBOOL_ERROR_HPP
