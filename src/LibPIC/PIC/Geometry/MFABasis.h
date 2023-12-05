/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/CartCoord.h>
#include <PIC/Config.h>
#include <PIC/CurviCoord.h>
#include <PIC/Predefined.h>
#include <PIC/VT/FourTensor.h>
#include <PIC/VT/FourVector.h>
#include <PIC/VT/Tensor.h>
#include <PIC/VT/Vector.h>

#include <cmath>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
class MFABasis {
    Real m_xi;
    Real m_xi2;
    Real m_D1;
    Real m_inv_D1;

protected:
    MFABasis(Real const xi, Vector const &D) noexcept
    : m_xi{ xi }, m_xi2{ xi * xi }, m_D1{ D.x }, m_inv_D1{ 1 / D.x }
    {
    }

private:
    [[nodiscard]] inline static auto pow2(Real const x) noexcept { return x * x; }

    [[nodiscard]] auto BOB0(CartCoord const &pos) const noexcept { return 1 + m_xi2 * pow2(pos.x); }
    [[nodiscard]] auto inv_sqrt_BOB0(CartCoord const &pos) const noexcept { return 1 / std::sqrt(BOB0(pos)); }
    [[nodiscard]] auto inv_sqrt_BOB0(CurviCoord const &pos) const noexcept { return std::cos(m_xi * m_D1 * pos.q1); }
    [[nodiscard]] auto BOB0(CurviCoord const &pos) const noexcept { return pow2(1 / inv_sqrt_BOB0(pos)); }

    template <class Coord>
    [[nodiscard]] ContrVector impl_Bcontr_div_B0(Coord const &) const noexcept { return { m_inv_D1, 0, 0 }; }
    template <class Coord>
    [[nodiscard]] CovarVector impl_Bcovar_div_B0(Coord const &pos) const noexcept { return { m_D1 * pow2(BOB0(pos)), 0, 0 }; }
    template <class Coord>
    [[nodiscard]] CartVector impl_Bcart_div_B0(Coord const &pos) const noexcept { return { BOB0(pos), 0, 0 }; }
    template <class Coord>
    [[nodiscard]] Real impl_Bmag_div_B0(Coord const &pos) const noexcept { return BOB0(pos); }

public:
    /// Contravariant components of B/B0
    ///
    [[nodiscard]] decltype(auto) Bcontr_div_B0(CartCoord const &pos) const noexcept { return impl_Bcontr_div_B0(pos); }
    [[nodiscard]] decltype(auto) Bcontr_div_B0(CurviCoord const &pos) const noexcept { return impl_Bcontr_div_B0(pos); }

    /// Covariant components of B/B0
    ///
    [[nodiscard]] decltype(auto) Bcovar_div_B0(CartCoord const &pos) const noexcept { return impl_Bcovar_div_B0(pos); }
    [[nodiscard]] decltype(auto) Bcovar_div_B0(CurviCoord const &pos) const noexcept { return impl_Bcovar_div_B0(pos); }

    /// Cartesian components of B/B0
    ///
    [[nodiscard]] decltype(auto) Bcart_div_B0(CartCoord const &pos) const noexcept { return impl_Bcart_div_B0(pos); }
    [[nodiscard]] decltype(auto) Bcart_div_B0(CurviCoord const &pos) const noexcept { return impl_Bcart_div_B0(pos); }

    /// Cartesian components of B/B0
    /// \param pos Cartesian x-component of position.
    /// \param pos_y Cartesian y-component of position.
    /// \param pos_z Cartesian z-component of position.
    ///
    [[nodiscard]] CartVector Bcart_div_B0(CartCoord const &pos, Real pos_y, Real pos_z) const noexcept
    {
        return { BOB0(pos), -m_xi2 * pos.x * pos_y, -m_xi2 * pos.x * pos_z };
    }

    /// Magnitude of B/B0
    ///
    [[nodiscard]] auto Bmag_div_B0(CartCoord const &pos) const noexcept { return impl_Bmag_div_B0(pos); }
    [[nodiscard]] auto Bmag_div_B0(CurviCoord const &pos) const noexcept { return impl_Bmag_div_B0(pos); }

    // MARK:- Basis
    template <class Coord>
    [[nodiscard]] static decltype(auto) e1(Coord const &) noexcept { return CartVector{ 1, 0, 0 }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) e2(Coord const &) noexcept { return CartVector{ 0, 1, 0 }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) e3(Coord const &) noexcept { return CartVector{ 0, 0, 1 }; }

    // MARK:- Vector Transform
    // for the present 1D situation, all mfa <-> cart vector transformations at the central field line are just pass-through
    template <class Coord>
    [[nodiscard]] static decltype(auto) mfa_to_cart(MFAVector const &vmfa, Coord const &) noexcept { return CartVector{ vmfa }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) mfa_to_cart(MFATensor const &Tmfa, Coord const &) noexcept { return CartTensor{ Tmfa }; }

    template <class Coord>
    [[nodiscard]] static decltype(auto) cart_to_mfa(CartVector const &vcart, Coord const &) noexcept { return MFAVector{ vcart }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) cart_to_mfa(CartTensor const &Tcart, Coord const &) noexcept { return MFATensor{ Tcart }; }

    template <class Coord>
    [[nodiscard]] static decltype(auto) mfa_to_cart(FourMFAVector const &vmfa, Coord const &) noexcept { return FourCartVector{ vmfa }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) mfa_to_cart(FourMFATensor const &Tmfa, Coord const &) noexcept { return FourCartTensor{ Tmfa }; }

    template <class Coord>
    [[nodiscard]] static decltype(auto) cart_to_mfa(FourCartVector const &vcart, Coord const &) noexcept { return FourMFAVector{ vcart }; }
    template <class Coord>
    [[nodiscard]] static decltype(auto) cart_to_mfa(FourCartTensor const &vvcart, Coord const &) noexcept { return FourMFATensor{ vvcart }; }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
