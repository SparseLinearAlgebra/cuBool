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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace cubool {

    template<typename Config>
    struct StreamsWrapper {
        StreamsWrapper() {
            for (auto& s : streams)
                cudaStreamCreate(&s);
        }

        ~StreamsWrapper() {
            for (auto& s : streams)
                cudaStreamDestroy(s);
        }

        cudaStream_t streams[Config::binsCount()] = {};
    };

    template<size_t Threads, size_t BlocksSize, size_t Max, size_t Min, size_t Id>
    struct Bin {
        static_assert(Threads <= BlocksSize, "Block size must be >= threads in this block");

        static constexpr size_t threads       = Threads;
        static constexpr size_t blockSize     = BlocksSize;
        static constexpr size_t dispatchRatio = BlocksSize / Threads;
        static constexpr size_t min           = Max;
        static constexpr size_t max           = Min;
        static constexpr size_t id            = Id;
    };


    template<typename... Bins>
    struct Config {
    public:
        static __host__ __device__ size_t selectBin(size_t rowSize) {
            static constexpr size_t mins[] = {Bins::min...};
            static constexpr size_t maxs[] = {Bins::max...};

            for (size_t i = 0; i < binsCount(); i++) {
                if (mins[i] <= rowSize && rowSize <= maxs[i])
                    return i;
            }

            return unusedBinId();
        }

        static __host__ __device__ constexpr size_t binBlockSize(size_t id) {
            constexpr size_t blockSizes[] = {Bins::blockSize...};
            return blockSizes[id];
        }

        static __host__ __device__ constexpr size_t binsCount() {
            return sizeof...(Bins);
        }

        static __host__ __device__ constexpr size_t unusedBinId() {
            return binsCount() + 1;
        }
    };

}// namespace cubool

#endif//CUBOOL_META_HPP
