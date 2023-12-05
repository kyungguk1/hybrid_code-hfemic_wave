/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "MirrorGeometry.h"
#include "../UTL/lippincott.h"

#include <cmath>

LIBPIC_NAMESPACE_BEGIN(1)
Detail::MirrorGeometry::MirrorGeometry(Real const xi, Vector const &D)
: CurviBasis{ xi, D }
, MFABasis{ xi, D }
, m_D{ D }
, m_xi{ xi }
, m_sqrt_g{ m_D.x * m_D.y * m_D.z }
, m_det_gij{ m_sqrt_g * m_sqrt_g }
, m_homogeneous{ xi < inhomogeneity_xi_threshold }
{
}

auto Detail::MirrorGeometry::impl_cart_to_curvi(CartCoord const &pos) const noexcept -> CurviCoord
{
    if (m_homogeneous) {
        constexpr auto sqrt_3 = 1.732050807568877193176604123436845839024;
        auto const     tmp    = m_xi * pos.x / sqrt_3;
        return CurviCoord{ (1 - tmp) * (1 + tmp) * pos.x / D1() };
    } else {
        return CurviCoord{ std::atan(xi() * pos.x) / (xi() * D1()) };
    }
}
auto Detail::MirrorGeometry::impl_curvi_to_cart(CurviCoord const &pos) const noexcept -> CartCoord
{
    if (m_homogeneous) {
        auto const D1q1 = pos.q1 * D1();
        return CartCoord{ D1q1 * (1 + m_xi * m_xi * D1q1 * D1q1 / 3) };
    } else {
        auto const xiD1q1 = xi() * D1() * pos.q1;
#if defined(DEBUG)
        if (std::abs(xiD1q1) >= M_PI_2)
            fatal_error("|xi*D1*q1| cannot be larger than pi/2");
#endif
        return CartCoord{ std::tan(xiD1q1) / xi() };
    }
}
LIBPIC_NAMESPACE_END(1)
