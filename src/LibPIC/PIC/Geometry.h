/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Geometry/MirrorGeometry.h>

LIBPIC_NAMESPACE_BEGIN(1)
class Geometry : public Detail::MirrorGeometry {
    Real m_O0;

public:
    Geometry() noexcept;
    Geometry(Real xi, Vector const &D, Real O0);

    /// Construct a Geometry object
    /// \param xi Mirror field inhomogeneity.
    /// \param D1 Grid scale factor along the parallel curvilinear coordinate.
    /// \param O0 The equatorial background magnetic field.
    ///
    Geometry(Real const xi, Real const D1, Real const O0)
    : Geometry{ xi, { D1, 1, 1 }, O0 } {}

    /// Magnitude of B at the origin
    [[nodiscard]] auto B0() const noexcept { return m_O0; }

    /// Magnitude of B/B0
    template <class Coord>
    [[nodiscard]] Real Bmag(Coord const &pos) const noexcept { return Bmag_div_B0(pos) * B0(); }

    /// Contravariant components of B
    template <class Coord>
    [[nodiscard]] decltype(auto) Bcontr(Coord const &pos) const noexcept { return Bcontr_div_B0(pos) * B0(); }

    /// Covariant components of B
    template <class Coord>
    [[nodiscard]] decltype(auto) Bcovar(Coord const &pos) const noexcept { return Bcovar_div_B0(pos) * B0(); }

    /// Cartesian components of B
    template <class Coord>
    [[nodiscard]] decltype(auto) Bcart(Coord const &pos) const noexcept { return Bcart_div_B0(pos) * B0(); }

    /// Cartesian components of B
    /// \tparam Coord Coordinate type.
    /// \param pos First component of position coordinates.
    /// \param pos_y Cartesian y-component of position.
    /// \param pos_z Cartesian z-component of position.
    template <class Coord>
    [[nodiscard]] decltype(auto) Bcart(Coord const &pos, Real pos_y, Real pos_z) const noexcept { return Bcart_div_B0(pos, pos_y, pos_z) * B0(); }
};
LIBPIC_NAMESPACE_END(1)
