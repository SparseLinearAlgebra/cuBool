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

#ifndef CUBOOL_META_HPP
#define CUBOOL_META_HPP

#include <cstddef>

namespace cubool {

    template <size_t workersCount, size_t blockSize>
    struct BinConfig {
        static const size_t mWorkersCount = workersCount;
        static const size_t mBlockSize = blockSize;
    };

    template <size_t minBorder, size_t maxBorder>
    struct BorderConfig {
        static const size_t mMinBorder = minBorder;
        static const size_t mMaxBorder = maxBorder;
    };

    template <typename ... BinsConfig>
    class ComputeConfig {
    public:

        template<typename T, typename C>
        void exec(C& context) {
            ExecImpl<void, T, BinsConfig...>::exec(0, context);
        }

    private:

        template<typename D, typename T, typename C, typename ... Bins>
        struct ExecImpl {
            static void exec(size_t index, C& context) { }
        };

        template<typename D, typename T, typename C, typename Bin>
        struct ExecImpl<D, T, C, Bin> {
            static void exec(size_t index, C& context) {
                using Timpl = typename T::template function<Bin>;

                Timpl timpl;
                timpl(index, context);
            }
        };

        template<typename D, typename T, typename C, typename Bin, typename ... Bins>
        struct ExecImpl<D, T, C, Bin, Bins...> {
            static void exec(size_t index, C& context) {
                using Timpl = typename T::template function<Bin>;

                Timpl timpl;
                timpl(index, context);

                ExecImpl<D, T, Bins...>::exec(index + 1, context);
            }
        };

    };

}

#endif //CUBOOL_META_HPP
