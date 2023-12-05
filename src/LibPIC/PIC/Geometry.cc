/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Geometry.h"

#include <limits>
#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
namespace {
constexpr auto quiet_nan = std::numeric_limits<Real>::quiet_NaN();
}
Geometry::Geometry() noexcept
: Detail::MirrorGeometry{ quiet_nan, { quiet_nan, quiet_nan, quiet_nan } }
, m_O0{ quiet_nan }
{
}
Geometry::Geometry(Real const xi, Vector const &D, Real const O0)
: MirrorGeometry{ xi, D }
, m_O0{ O0 }
{
    if (xi < 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - negative xi" };
    if (D.x <= 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - non-positive D1" };
    if (D.y <= 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - non-positive D2" };
    if (D.z <= 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - non-positive D3" };
    if (O0 <= 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - non-positive O0" };
}
LIBPIC_NAMESPACE_END(1)
