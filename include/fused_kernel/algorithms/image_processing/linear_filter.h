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

#ifndef FK_LINEAR_FILTER
#define FK_LINEAR_FILTER

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>

namespace fk {

    // Border policy for out-of-image neighbours (mirrors NppiBorderType subset).
    enum class FilterBorder { CONSTANT_ZERO, REPLICATE };

    // Rounding applied when casting the float accumulator back to an integer T.
    // NEAREST = round-half-to-even (via saturate_cast); TRUNCATE = round-toward-zero.
    // NPP's FilterBox truncates; general FilterBorder convolution on float needs no
    // rounding. Pick TRUNCATE to match nppiFilterBox*.
    enum class FilterRounding { NEAREST, TRUNCATE };

    // Linear 2D filter (general convolution): out = round( sum_k(src_k * kernel_k) ).
    // The kernel is a float coefficient buffer of size kW x kH with an anchor.
    // Box filter = uniform kernel 1/(kW*kH) + TRUNCATE; Gaussian = sampled coeffs;
    // general convolution = arbitrary coeffs. Arithmetic is done in float and the
    // result cast back to T.
    template <ND D, typename T>
    struct LinearFilterParams {
        RawPtr<D, T> src;
        RawPtr<ND::_2D, float> kernel; // coefficients
        int kW;
        int kH;
        int anchorX;
        int anchorY;
    };

    template <ND D, typename T, FilterBorder BORDER = FilterBorder::REPLICATE,
              FilterRounding ROUND = FilterRounding::NEAREST>
    struct LinearFilter {
    private:
        using Parent = ReadOperation<T, LinearFilterParams<D, T>, T, TF::DISABLED, LinearFilter<D, T, BORDER, ROUND>>;
        using SelfType = LinearFilter<D, T, BORDER, ROUND>;
        using FloatVec = VectorType_t<float, cn<T>>;
    public:
        FK_STATIC_STRUCT(LinearFilter, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> T {
            const int W = static_cast<int>(params.src.dims.width);
            const int H = static_cast<int>(params.src.dims.height);
            FloatVec acc = make_set<FloatVec>(0.f);
            for (int ky = 0; ky < params.kH; ++ky) {
                for (int kx = 0; kx < params.kW; ++kx) {
                    const float coeff = params.kernel.data[ky * (params.kernel.dims.pitch / (int)sizeof(float)) + kx];
                    int sx = static_cast<int>(thread.x) + kx - params.anchorX;
                    int sy = static_cast<int>(thread.y) + ky - params.anchorY;
                    T v;
                    if constexpr (BORDER == FilterBorder::REPLICATE) {
                        sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
                        sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
                        v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                    } else {
                        if (sx < 0 || sx >= W || sy < 0 || sy >= H) {
                            v = make_set<T>(static_cast<VBase<T>>(0));
                        } else {
                            v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                        }
                    }
                    acc = acc + (toFloatVec(v) * coeff);
                }
            }
            if constexpr (ROUND == FilterRounding::TRUNCATE && std::is_integral_v<VBase<T>>) {
                return truncCast(acc);
            } else {
                return cxp::saturate_cast<T>::f(acc);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.src.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.src.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return 1;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.src.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), 1u };
        }
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& src, const Ptr<ND::_2D, float>& kernel,
                                            int kW, int kH, int anchorX, int anchorY) {
            return { { LinearFilterParams<D, T>{ src.ptr(), kernel.ptr(), kW, kH, anchorX, anchorY } } };
        }
    private:
        FK_HOST_DEVICE_FUSE FloatVec toFloatVec(const T& v) {
            if constexpr (cn<T> == 1) {
                return make_set<FloatVec>(static_cast<float>(v));
            } else {
                return cxp::cast<FloatVec>::f(v);
            }
        }
        // Truncate toward zero with clamp to T's range (matches NPP FilterBox).
        FK_HOST_DEVICE_FUSE T truncCast(const FloatVec& acc) {
            using Base = VBase<T>;
            constexpr float lo = static_cast<float>(minValue<Base>);
            constexpr float hi = static_cast<float>(maxValue<Base>);
            auto clampTrunc = [](float f) -> Base {
                float t = f < 0.f ? -cxp::floor::f(-f) : cxp::floor::f(f); // toward zero
                t = t < lo ? lo : (t > hi ? hi : t);
                return static_cast<Base>(t);
            };
            if constexpr (cn<T> == 1) {
                return clampTrunc(acc);
            } else if constexpr (cn<T> == 2) {
                return T{ clampTrunc(acc.x), clampTrunc(acc.y) };
            } else if constexpr (cn<T> == 3) {
                return T{ clampTrunc(acc.x), clampTrunc(acc.y), clampTrunc(acc.z) };
            } else {
                return T{ clampTrunc(acc.x), clampTrunc(acc.y), clampTrunc(acc.z), clampTrunc(acc.w) };
            }
        }
    };

    // Box (mean) filter: averages a kW x kH window. Accumulates the window sum in
    // float and divides by the area ONCE (sum-then-divide), then truncates toward
    // zero — matching nppiFilterBox / nppiFilterBoxBorder. Doing the divide once
    // avoids the per-coefficient rounding error of the general convolution path.
    template <ND D, typename T, FilterBorder BORDER = FilterBorder::REPLICATE>
    struct BoxFilter {
    private:
        using Parent = ReadOperation<T, LinearFilterParams<D, T>, T, TF::ENABLED, BoxFilter<D, T, BORDER>>;
        using SelfType = BoxFilter<D, T, BORDER>;
        using FloatVec = VectorType_t<float, cn<T>>;
        static constexpr int MAX_COLS = 64; // cap for the per-column-sum scratch (ELEMS + kW - 1)
    public:
        FK_STATIC_STRUCT(BoxFilter, SelfType)
        DECLARE_READ_PARENT

        // Scalar path (1 px): used by the non-thread-fused executor path.
        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> T {
            return filterAt(static_cast<int>(thread.x), static_cast<int>(thread.y), thread.z, params);
        }

        // Thread-fused path: compute ELEMS_PER_THREAD horizontally-contiguous output
        // pixels per thread and return the packed vector type the executor stores
        // coalesced. The group's first output x is thread.x*ELEMS (the executor passes
        // the group index in thread.x and reads/writes a wide type, matching
        // PerThreadRead/Write). More work per thread hides latency — what NPP's "Quad"
        // small-filter kernels do. For single-channel interior groups we further reuse
        // the overlapping per-column vertical sums across the ELEMS outputs.
        template <uint ELEMS_PER_THREAD>
        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params)
            -> ThreadFusionType<T, ELEMS_PER_THREAD, T> {
            if constexpr (ELEMS_PER_THREAD == 1) {
                return filterAt(static_cast<int>(thread.x), static_cast<int>(thread.y), thread.z, params);
            } else {
                using WideType = ThreadFusionType<T, ELEMS_PER_THREAD, T>;
                const int E = static_cast<int>(ELEMS_PER_THREAD);
                const int baseOutX = static_cast<int>(thread.x) * E;
                const int oy = static_cast<int>(thread.y);
                const int W = static_cast<int>(params.src.dims.width);
                const int H = static_cast<int>(params.src.dims.height);
                const int firstWinX = baseOutX - params.anchorX;
                const int lastWinX  = (baseOutX + E - 1) - params.anchorX + params.kW - 1;
                const bool interiorAll = (firstWinX >= 0) && (lastWinX < W) &&
                                         (oy - params.anchorY >= 0) && (oy - params.anchorY + params.kH - 1 < H);
                WideType out;
                T* o = reinterpret_cast<T*>(&out);
                if constexpr (cn<T> == 1) {
                    if (interiorAll && (E + params.kW - 1) <= MAX_COLS) {
                        const int baseY = oy - params.anchorY;
                        const int nCols = E + params.kW - 1;
                        float colSum[MAX_COLS];
                        for (int c = 0; c < nCols; ++c) {
                            float s = 0.f;
                            const int sx = firstWinX + c;
                            for (int ky = 0; ky < params.kH; ++ky) {
                                const T v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, baseY + ky, thread.z}, params.src);
                                s += static_cast<float>(v);
                            }
                            colSum[c] = s;
                        }
                        const float area = static_cast<float>(params.kW * params.kH);
                        #pragma unroll
                        for (int e = 0; e < E; ++e) {
                            float sum = 0.f;
                            for (int kx = 0; kx < params.kW; ++kx) sum += colSum[e + kx];
                            float q = sum / area;
                            float t = q < 0.f ? -cxp::floor::f(-q) : cxp::floor::f(q);
                            o[e] = static_cast<VBase<T>>(t);
                        }
                    } else {
                        #pragma unroll
                        for (int e = 0; e < E; ++e) o[e] = filterAt(baseOutX + e, oy, thread.z, params);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < E; ++e) o[e] = filterAt(baseOutX + e, oy, thread.z, params);
                }
                return out;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.width; }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.height; }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) { return 1; }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) { return opData.params.src.dims.pitch; }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), 1u };
        }
        // anchor defaults to mask centre; pass explicitly to match NPP.
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& src, int kW, int kH, int anchorX, int anchorY) {
            return { { LinearFilterParams<D, T>{ src.ptr(), RawPtr<ND::_2D, float>{}, kW, kH, anchorX, anchorY } } };
        }
    private:
        // Box filter value at a single output pixel (ox, oy), with interior fast path.
        FK_HOST_DEVICE_FUSE T filterAt(int ox, int oy, int z, const ParamsType& params) {
            const int W = static_cast<int>(params.src.dims.width);
            const int H = static_cast<int>(params.src.dims.height);
            FloatVec sum = make_set<FloatVec>(0.f);
            const bool interior = (ox - params.anchorX >= 0) && (ox - params.anchorX + params.kW - 1 < W) &&
                                  (oy - params.anchorY >= 0) && (oy - params.anchorY + params.kH - 1 < H);
            if (interior) {
                const int baseX = ox - params.anchorX;
                const int baseY = oy - params.anchorY;
                for (int ky = 0; ky < params.kH; ++ky) {
                    for (int kx = 0; kx < params.kW; ++kx) {
                        const T v = *PtrAccessor<D>::template cr_point<T, T>(Point{baseX + kx, baseY + ky, z}, params.src);
                        sum = sum + toFloatVecB(v);
                    }
                }
            } else {
                for (int ky = 0; ky < params.kH; ++ky) {
                    for (int kx = 0; kx < params.kW; ++kx) {
                        int sx = ox + kx - params.anchorX;
                        int sy = oy + ky - params.anchorY;
                        T v;
                        if constexpr (BORDER == FilterBorder::REPLICATE) {
                            sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
                            sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
                            v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, z}, params.src);
                        } else {
                            if (sx < 0 || sx >= W || sy < 0 || sy >= H) {
                                v = make_set<T>(static_cast<VBase<T>>(0));
                            } else {
                                v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, z}, params.src);
                            }
                        }
                        sum = sum + toFloatVecB(v);
                    }
                }
            }
            const float area = static_cast<float>(params.kW * params.kH);
            return divTruncCast(sum, area);
        }
        FK_HOST_DEVICE_FUSE FloatVec toFloatVecB(const T& v) {
            if constexpr (cn<T> == 1) return make_set<FloatVec>(static_cast<float>(v));
            else return cxp::cast<FloatVec>::f(v);
        }
        FK_HOST_DEVICE_FUSE T divTruncCast(const FloatVec& sum, float area) {
            using Base = VBase<T>;
            constexpr float lo = static_cast<float>(minValue<Base>);
            constexpr float hi = static_cast<float>(maxValue<Base>);
            auto f1 = [area](float s) -> Base {
                float q = s / area;
                float t = q < 0.f ? -cxp::floor::f(-q) : cxp::floor::f(q);
                t = t < lo ? lo : (t > hi ? hi : t);
                return static_cast<Base>(t);
            };
            if constexpr (cn<T> == 1) return f1(sum);
            else if constexpr (cn<T> == 2) return T{ f1(sum.x), f1(sum.y) };
            else if constexpr (cn<T> == 3) return T{ f1(sum.x), f1(sum.y), f1(sum.z) };
            else return T{ f1(sum.x), f1(sum.y), f1(sum.z), f1(sum.w) };
        }
    };

} // namespace fk

#endif
