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

#ifndef FK_MORPHOLOGY
#define FK_MORPHOLOGY

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {

    enum class MorphologyType { ERODE, DILATE };

    // Border policy for out-of-image neighbours, matching NppiBorderType.
    enum class MorphBorder { CONSTANT_ZERO, REPLICATE };

    // Structuring-element parameters: source image, mask (1 = active), mask size,
    // and anchor (the mask cell that aligns with the output pixel).
    template <ND D, typename T>
    struct MorphologyParams {
        RawPtr<D, T> src;
        RawPtr<ND::_2D, uchar> mask;
        int maskW;
        int maskH;
        int anchorX;
        int anchorY;
    };

    // Erosion (min) / Dilation (max) over the structuring element.
    // Border handling is selectable: CONSTANT_ZERO matches nppiErode/Dilate's
    // constant-zero edge behaviour; REPLICATE clamps neighbour coordinates to the
    // image edge, matching nppiErodeBorder/nppiDilateBorder with NPP_BORDER_REPLICATE.
    template <ND D, typename T, MorphologyType MT, MorphBorder BORDER = MorphBorder::REPLICATE>
    struct Morphology {
    private:
        using Parent = ReadOperation<T, MorphologyParams<D, T>, T, TF::DISABLED, Morphology<D, T, MT, BORDER>>;
        using SelfType = Morphology<D, T, MT, BORDER>;
    public:
        FK_STATIC_STRUCT(Morphology, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> T {
            const int W = static_cast<int>(params.src.dims.width);
            const int H = static_cast<int>(params.src.dims.height);
            using Base = VBase<T>;
            T acc = make_set<T>(static_cast<Base>(0));
            bool any = false;
            for (int my = 0; my < params.maskH; ++my) {
                for (int mx = 0; mx < params.maskW; ++mx) {
                    if (params.mask.data[my * params.mask.dims.pitch + mx] == 0) continue;
                    int sx = static_cast<int>(thread.x) + mx - params.anchorX;
                    int sy = static_cast<int>(thread.y) + my - params.anchorY;
                    T v;
                    if constexpr (BORDER == MorphBorder::REPLICATE) {
                        sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
                        sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
                        v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                    } else {
                        if (sx < 0 || sx >= W || sy < 0 || sy >= H) {
                            v = make_set<T>(static_cast<Base>(0));
                        } else {
                            v = *PtrAccessor<D>::template cr_point<T, T>(Point{sx, sy, thread.z}, params.src);
                        }
                    }
                    if (!any) { acc = v; any = true; }
                    else if constexpr (MT == MorphologyType::ERODE) { acc = cxp::min::f(acc, v); }
                    else { acc = cxp::max::f(acc, v); }
                }
            }
            return acc;
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
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& src, const Ptr<ND::_2D, uchar>& mask,
                                            int maskW, int maskH, int anchorX, int anchorY) {
            return { { MorphologyParams<D, T>{ src.ptr(), mask.ptr(), maskW, maskH, anchorX, anchorY } } };
        }
    };

    template <ND D, typename T, MorphBorder BORDER = MorphBorder::REPLICATE>
    using Erode = Morphology<D, T, MorphologyType::ERODE, BORDER>;
    template <ND D, typename T, MorphBorder BORDER = MorphBorder::REPLICATE>
    using Dilate = Morphology<D, T, MorphologyType::DILATE, BORDER>;

} // namespace fk

#endif
