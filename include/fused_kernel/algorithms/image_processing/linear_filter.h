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
        using Parent = ReadOperation<T, LinearFilterParams<D, T>, T, TF::DISABLED, BoxFilter<D, T, BORDER>>;
        using SelfType = BoxFilter<D, T, BORDER>;
        using FloatVec = VectorType_t<float, cn<T>>;
    public:
        FK_STATIC_STRUCT(BoxFilter, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> T {
            const int W = static_cast<int>(params.src.dims.width);
            const int H = static_cast<int>(params.src.dims.height);
            const int tx = static_cast<int>(thread.x);
            const int ty = static_cast<int>(thread.y);
            FloatVec sum = make_set<FloatVec>(0.f);
            // Interior fast path: when the whole window lies inside the image, no
            // per-tap border clamping is needed. This is the common case and avoids
            // ~4 compares/selects per tap (the dominant cost for small kernels).
            const bool interior = (tx - params.anchorX >= 0) && (tx - params.anchorX + params.kW - 1 < W) &&
                                  (ty - params.anchorY >= 0) && (ty - params.anchorY + params.kH - 1 < H);
            if (interior) {
                const int baseX = tx - params.anchorX;
                const int baseY = ty - params.anchorY;
                for (int ky = 0; ky < params.kH; ++ky) {
                    for (int kx = 0; kx < params.kW; ++kx) {
                        const T v = *PtrAccessor<D>::template cr_point<T, T>(Point{baseX + kx, baseY + ky, thread.z}, params.src);
                        sum = sum + toFloatVecB(v);
                    }
                }
            } else {
                for (int ky = 0; ky < params.kH; ++ky) {
                    for (int kx = 0; kx < params.kW; ++kx) {
                        int sx = tx + kx - params.anchorX;
                        int sy = ty + ky - params.anchorY;
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
                        sum = sum + toFloatVecB(v);
                    }
                }
            }
            const float area = static_cast<float>(params.kW * params.kH);
            return divTruncCast(sum, area);
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
