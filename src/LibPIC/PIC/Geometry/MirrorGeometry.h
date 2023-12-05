/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Geometry/CurviBasis.h>
#include <PIC/Geometry/MFABasis.h>

#include <cmath>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
class MirrorGeometry
: public CurviBasis
, public MFABasis {
    static constexpr Real inhomogeneity_xi_threshold = 1e-10;

    Vector m_D;
    Real   m_xi;
    Real   m_sqrt_g;
    Real   m_det_gij;
    bool   m_homogeneous;

    [[nodiscard]] auto impl_cart_to_curvi(CartCoord const &) const noexcept -> CurviCoord;
    [[nodiscard]] auto impl_curvi_to_cart(CurviCoord const &) const noexcept -> CartCoord;

protected:
    MirrorGeometry(Real xi, Vector const &D);

public:
    // properties
    [[nodiscard]] Real xi() const noexcept { return m_xi; }
    [[nodiscard]] bool is_homogeneous() const noexcept { return m_homogeneous; }

    [[nodiscard]] Vector D() const noexcept { return m_D; }
    [[nodiscard]] Real   D1() const noexcept { return m_D.x; }
    [[nodiscard]] Real   D2() const noexcept { return m_D.y; }
    [[nodiscard]] Real   D3() const noexcept { return m_D.z; }

    // √g
    [[nodiscard]] Real sqrt_g() const noexcept { return m_sqrt_g; }
    // g = det(g_ij)
    [[nodiscard]] Real det_gij() const noexcept { return m_det_gij; }

    /// Check if the parallel curvilinear coordinate is within the valid range
    /// \param pos Curvilinear coordinate.
    ///
    [[nodiscard]] bool is_valid(CurviCoord const &pos) const noexcept { return std::abs(xi() * D1() * pos.q1) < M_PI_2; }

    /// From Cartesian to curvilinear coordinate transformation
    /// \param pos Cartesian coordinates.
    /// \return Curvilinear coordinates.
    [[nodiscard]] CurviCoord cotrans(CartCoord const &pos) const noexcept { return impl_cart_to_curvi(pos); };

    /// From curvilinear to Cartesian coordinate transformation
    /// \param pos Curvilinear coordinates.
    /// \return Cartesian coordinates.
    [[nodiscard]] CartCoord cotrans(CurviCoord const &pos) const noexcept { return impl_curvi_to_cart(pos); };

    /// Cartesian components of B/B0
    /// \param pos Curvilinear q1-component of position.
    /// \param pos_y Cartesian y-component of position.
    /// \param pos_z Cartesian z-component of position.
    ///
    [[nodiscard]] CartVector Bcart_div_B0(CurviCoord const &pos, Real pos_y, Real pos_z) const noexcept
    {
        return Bcart_div_B0(cotrans(pos), pos_y, pos_z);
    }
    using MFABasis::Bcart_div_B0;
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
