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

#ifndef CUBOOL_LOGGER_HPP
#define CUBOOL_LOGGER_HPP

#include <string>
#include <vector>
#include <functional>
#include <sstream>

namespace cubool {

    /**
     * Basic text logger.
     */
    class Logger {
    public:
        enum class Level {
            Info,       // Logged basic info (can be ignored)
            Warning,    // Logged warning (can be ignored)
            Error,      // Logged error (can be ignored)
            Always      // Logged always (cannot be ignored)
        };

        virtual ~Logger() = default;
        virtual void log(Level level, const std::string &message) = 0;
        virtual void logInfo(const std::string &message);
        virtual void logWarning(const std::string &message);
        virtual void logError(const std::string &message);
        virtual bool isDummy() const = 0;
        virtual size_t getMessagesCount() const = 0;
    };

    /**
     * @brief Text logger
     *
     * Collects logged messages into an array.
     * Allows add filters to ignore some messages.
     * Allows commit messages to the output log file.
     */
    class TextLogger final: public Logger {
    public:
        /** Allows filter logged messages (before they are actually saved inside) */
        using Filter = std::function<bool(Level level, const std::string& message)>;
        /** Allows perform post-logging action (if message was saved) */
        using OnLogged = std::function<void(size_t id, Level level, const std::string& message)>;

        ~TextLogger() override = default;
        void log(Level level, const std::string &message) override;
        size_t getMessagesCount() const override;
        bool isDummy() const override;

        void addFilter(Filter filter);
        void removeAllFilters();
        void addOnLoggerAction(OnLogged onLogged);
        void removeAllOnLoggedActions();

    private:
        struct Entry {
            std::string message;
            Level level;
            size_t id;
        };

        std::vector<Entry> mEntries;
        std::vector<Filter> mFilters;
        std::vector<OnLogged> mOnLogged;
        size_t mNextMessageId = 0;
    };

    /**
     * No logging logic.
     */
    class DummyLogger final: public Logger {
    public:
        DummyLogger() noexcept = default;
        ~DummyLogger() override = default;
        void log(Level level, const std::string &message) override;
        size_t getMessagesCount() const override;
        bool isDummy() const override;
    };

    /**
     * Utility to build and log fancy messages.
     */
    class LogStream {
    public:
        struct Commit {};
        constexpr static const Commit cmt = Commit{};

        explicit LogStream(Logger& logger)
            : mLogger(logger), mLevel(Logger::Level::Info) {
        };

        LogStream(LogStream&& other) noexcept = default;
        ~LogStream() = default;

        void commit() {
            if (!mLogger.isDummy()) {
                mLogger.log(mLevel, mStream.str());
                mStream.str(std::string());
            }
        }

        friend LogStream& operator<<(LogStream& stream, Logger::Level level) {
            if (!stream.mLogger.isDummy()) {
                stream.mLevel = level;
            }

            return stream;
        }

        friend LogStream& operator<<(LogStream& stream, Commit commit) {
            if (!stream.mLogger.isDummy()) {
                stream.commit();
            }

            return stream;
        }

        template<typename T>
        friend LogStream& operator<<(LogStream& stream, T&& t) {
            if (!stream.mLogger.isDummy()) {
                stream.mStream << std::forward<T>(t);
            }

            return stream;
        }

    private:
        Logger& mLogger;
        Logger::Level mLevel;
        std::stringstream mStream;
    };

}

#endif //CUBOOL_LOGGER_HPP
