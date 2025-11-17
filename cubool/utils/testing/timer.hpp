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

#ifndef CUBOOL_TESTING_TIMER_HPP
#define CUBOOL_TESTING_TIMER_HPP

#include <chrono>

namespace testing {

    struct Timer {
    public:
        void start() {
            mStart = mEnd = clock::now();
        }

        void end() {
            mEnd = clock::now();
        }

        double getElapsedTimeMs() const {
            using namespace std::chrono;
            return (double) duration_cast<nanoseconds>(mEnd - mStart).count() / 1.0e6;
        }

    private:
        using clock     = std::chrono::high_resolution_clock;
        using timepoint = clock::time_point;
        timepoint mStart;
        timepoint mEnd;
    };

    struct TimeQuery {
    public:
        void addTimeSample(double ms) {
            mSamplesCount += 1;
            mTimeSumMS += ms;
        }

        double getAverageTimeMs() const {
            return mTimeSumMS / (double) mSamplesCount;
        }

    private:
        double mTimeSumMS    = 0.0f;
        int    mSamplesCount = 0;
    };

    struct TimeScope {
    public:
        explicit TimeScope(TimeQuery& query) : mTimeQuery(query) {
            mTimer.start();
        }

        ~TimeScope() {
            mTimer.end();
            mTimeQuery.addTimeSample(mTimer.getElapsedTimeMs());
        }

    private:
        Timer      mTimer;
        TimeQuery& mTimeQuery;
    };

}// namespace testing

#endif//CUBOOL_TESTING_TIMER_HPP
