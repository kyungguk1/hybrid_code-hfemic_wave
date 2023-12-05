/*
 * Copyright (c) 2022-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "ExternalSource.h"

#include <algorithm>
#include <cmath>

HYBRID1D_BEGIN_NAMESPACE
template <unsigned N>
ExternalSource::ExternalSource(ParamSet const &params, ExternalSourceDesc<N> const &src)
: Species{ params }, src_desc{ src }, src_pos{ begin(src.pos), end(src.pos) }, src_Jre(N), src_Jim(N), number_of_source_points(N)
{
    src_desc.Oc = params.O0; // this is for CAM
    src_desc.op = 0;

    std::transform(begin(src.J0), end(src.J0), begin(src_Jre), [](auto const &cv) noexcept -> MFAVector {
        return { cv.x.real(), cv.y.real(), cv.z.real() };
    });
    std::transform(begin(src.J0), end(src.J0), begin(src_Jim), [](auto const &cv) noexcept -> MFAVector {
        return { cv.x.imag(), cv.y.imag(), cv.z.imag() };
    });

    // ramp slopes
    constexpr auto eps = 1e-15;
    (ramp_slope.ease_in = M_PI) /= src_desc.ease.in > eps ? src_desc.ease.in : 1.0;
    (ramp_slope.ease_out = M_PI) /= src_desc.ease.out > eps ? src_desc.ease.out : 1.0;
}
HYBRID1D_END_NAMESPACE
