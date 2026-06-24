/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MEDIAN_FILTER
#define FK_MEDIAN_FILTER

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h> // FilterBorder enum

namespace fk {

    // Median filter over a kW x kH window. Per channel, collects the window values,
    // sorts them, and returns the middle element — matching nppiFilterMedianBorder.
    // MAX_WINDOW caps the per-thread scratch array (compile-time); kW*kH must be <= it.
    template <ND D, typename T, int MAX_WINDOW = 49, FilterBorder BORDER = FilterBorder::REPLICATE>
    struct MedianFilter {
    private:
        using Parent = ReadOperation<T, LinearFilterParams<D, T>, T, TF::DISABLED, MedianFilter<D, T, MAX_WINDOW, BORDER>>;
        using SelfType = MedianFilter<D, T, MAX_WINDOW, BORDER>;
        using Base = VBase<T>;
        static constexpr int CH = cn<T>;
    public:
        FK_STATIC_STRUCT(MedianFilter, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> T {
            const int W = static_cast<int>(params.src.dims.width);
            const int H = static_cast<int>(params.src.dims.height);
            const int count = params.kW * params.kH;
            Base buf[CH][MAX_WINDOW];
            int n = 0;
            for (int ky = 0; ky < params.kH; ++ky) {
                for (int kx = 0; kx < params.kW; ++kx) {
                    int sx = static_cast<int>(thread.x) + kx - params.anchorX;
                    int sy = static_cast<int>(thread.y) + ky - params.anchorY;
                    T v;
                    if constexpr (BORDER == FilterBorder::REPLICATE) {
                        sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
                        sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
                        v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                    } else {
                        if (sx < 0 || sx >= W || sy < 0 || sy >= H) v = make_set<T>(static_cast<Base>(0));
                        else v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                    }
                    storeChannels(buf, n, v);
                    ++n;
                }
            }
            // Insertion sort each channel, then pick the middle element.
            T out;
            const int mid = count / 2;
            for (int c = 0; c < CH; ++c) {
                for (int i = 1; i < count; ++i) {
                    Base key = buf[c][i];
                    int j = i - 1;
                    while (j >= 0 && buf[c][j] > key) { buf[c][j + 1] = buf[c][j]; --j; }
                    buf[c][j + 1] = key;
                }
                setChannel(out, c, buf[c][mid]);
            }
            return out;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.width; }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.height; }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) { return 1; }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.pitch; }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), 1u };
        }
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& src, int kW, int kH, int anchorX, int anchorY) {
            return { { LinearFilterParams<D, T>{ src.ptr(), RawPtr<ND::_2D, float>{}, kW, kH, anchorX, anchorY } } };
        }
    private:
        FK_HOST_DEVICE_FUSE void storeChannels(Base buf[CH][MAX_WINDOW], int n, const T& v) {
            if constexpr (CH == 1) { buf[0][n] = v; }
            else if constexpr (CH == 2) { buf[0][n] = v.x; buf[1][n] = v.y; }
            else if constexpr (CH == 3) { buf[0][n] = v.x; buf[1][n] = v.y; buf[2][n] = v.z; }
            else { buf[0][n] = v.x; buf[1][n] = v.y; buf[2][n] = v.z; buf[3][n] = v.w; }
        }
        FK_HOST_DEVICE_FUSE void setChannel(T& out, int c, Base val) {
            if constexpr (CH == 1) { out = val; }
            else { (&out.x)[c] = val; }
        }
    };

} // namespace fk

#endif
