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

#include <io/logger.hpp>

namespace cubool {

    void Logger::logInfo(const std::string &message) {
        this->log(Level::Info, message);
    }

    void Logger::logWarning(const std::string &message) {
        this->log(Level::Warning, message);
    }

    void Logger::logError(const std::string &message) {
        this->log(Level::Error, message);
    }

    void TextLogger::log(Logger::Level level, const std::string &message) {
        bool pass = true;

        // If pass all filters
        for (const auto& filter: mFilters) {
            pass = pass && filter(level, message);
        }

        if (pass || level == Level::Always) {
            auto id = mNextMessageId++;

            Entry entry;
            entry.message = message;
            entry.level = level;
            entry.id = id;

            mEntries.emplace_back(std::move(entry));

            // Notify listeners
            for (const auto& action: mOnLogged) {
                action(id, level, message);
            }
        }
    }

    size_t TextLogger::getMessagesCount() const {
        return mEntries.size();
    }

    void TextLogger::addFilter(Filter filter) {
        mFilters.emplace_back(std::move(filter));
    }

    void TextLogger::removeAllFilters() {
        mFilters.clear();
    }

    void TextLogger::addOnLoggerAction(OnLogged onLogged) {
        mOnLogged.emplace_back(std::move(onLogged));
    }

    void TextLogger::removeAllOnLoggedActions() {
        mOnLogged.clear();
    }

    void DummyLogger::log(Logger::Level level, const std::string &message) {
        // no op.
    }

    size_t DummyLogger::getMessagesCount() const {
        return 0;
    }
}
