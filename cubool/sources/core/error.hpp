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
#include <sstream>

namespace cubool {

    /**
     * Generic library exception.
     * Use this one in particular backend implementations.
     */
    class Exception: public std::exception {
    public:

        Exception(std::string message, std::string function, std::string file, size_t line, cuBool_Status status, bool critical)
                : std::exception(),
                  mMessage(std::move(message)),
                  mFunction(std::move(function)),
                  mFile(std::move(file)),
                  mLine(line),
                  nStatus(status),
                  mCritical(critical) {

        }

        Exception(const Exception& e) noexcept = default;
        Exception(Exception&& e) noexcept = default;
        ~Exception() noexcept override = default;

        const char* what() const noexcept override {
            if (!mCached) {
                mCached = true;

                std::stringstream s;
                s << "\"" << getMessage() << "\" in" << std::endl
                  << getFile() << ": line: " << getLine() << " function: " << getFunction();

                mWhatCached = s.str();
            }

            return mWhatCached.c_str();
        }

        const std::string& getMessage() const noexcept {
            return mMessage;
        }

        const std::string& getFunction() const noexcept {
            return mFunction;
        }

        const std::string& getFile() const noexcept {
            return mFile;
        }

        size_t getLine() const {
            return mLine;
        }

        cuBool_Status getStatus() const noexcept {
            return nStatus;
        }

        bool isCritical() const noexcept {
            return mCritical;
        }

    private:
        mutable std::string mWhatCached;
        std::string mMessage;
        std::string mFunction;
        std::string mFile;
        size_t mLine;
        cuBool_Status nStatus;
        bool mCritical;
        mutable bool mCached = false;
    };

    /**
     * Exceptions with cuBool_Status error code parametrisation.
     * @tparam Type Exception error code (type)
     */
    template<cuBool_Status Type>
    class TException: public Exception {
    public:
        TException(std::string message, std::string&& function, std::string&& file, size_t line, bool critical)
                : Exception(std::move(message), std::move(function), std::move(file), line, Type, critical)  {

        }

        TException(const TException& other) noexcept = default;
        TException(TException&& other) noexcept = default;
        ~TException() noexcept override = default;
    };

    // Errors exposed to the C API
    using Error = TException<cuBool_Status::CUBOOL_STATUS_ERROR>;
    using DeviceError = TException<cuBool_Status::CUBOOL_STATUS_DEVICE_ERROR>;
    using DeviceNotPresent = TException<cuBool_Status::CUBOOL_STATUS_DEVICE_NOT_PRESENT>;
    using MemOpFailed = TException<cuBool_Status::CUBOOL_STATUS_MEM_OP_FAILED>;
    using InvalidArgument = TException<cuBool_Status::CUBOOL_STATUS_INVALID_ARGUMENT>;
    using InvalidState = TException<cuBool_Status::CUBOOL_STATUS_INVALID_STATE>;
    using BackendError = TException<cuBool_Status::CUBOOL_STATUS_BACKEND_ERROR>;
    using NotImplemented = TException<cuBool_Status::CUBOOL_STATUS_NOT_IMPLEMENTED>;

}

// An error, in theory, can recover after this
#define RAISE_ERROR(type, message)                                                      \
    do {                                                                                \
        throw ::cubool::type(message, __FUNCTION__, __FILE__, __LINE__, false);         \
    } while (0);

#define CHECK_RAISE_ERROR(condition, type, message)                                     \
    if (!(condition)) { RAISE_ERROR(type, #condition ": " message); } else { }

// Critical errors, cause library shutdown
#define RAISE_CRITICAL_ERROR(type, message)                                             \
    do {                                                                                \
        throw ::cubool::type(message, __FUNCTION__, __FILE__, __LINE__, true);          \
    } while (0);

#define CHECK_RAISE_CRITICAL_ERROR(condition, type, message)                            \
    if (!(condition)) { RAISE_CRITICAL_ERROR(type, #condition ": " message); } else { }

#endif //CUBOOL_ERROR_HPP
