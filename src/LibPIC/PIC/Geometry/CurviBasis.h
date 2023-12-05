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
#include <PIC/VT/Tensor.h>
#include <PIC/VT/Vector.h>

#include <cmath>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
class CurviBasis {
    Real m_xi;
    Real m_xi2;
    Real m_D1;
    Real m_D2;
    Real m_D3;
    Real m_inv_D1;
    Real m_inv_D2;
    Real m_inv_D3;

protected:
    CurviBasis(Real const xi, Vector const &D) noexcept
    : m_xi{ xi }, m_xi2{ xi * xi }, m_D1{ D.x }, m_D2{ D.y }, m_D3{ D.z }, m_inv_D1{ 1 / D.x }, m_inv_D2{ 1 / D.y }, m_inv_D3{ 1 / D.z }
    {
    }

private:
    [[nodiscard]] inline static auto pow2(Real const x) noexcept { return x * x; }

    [[nodiscard]] auto BOB0(CartCoord const &pos) const noexcept { return 1 + m_xi2 * pow2(pos.x); }
    [[nodiscard]] auto inv_sqrt_BOB0(CartCoord const &pos) const noexcept { return 1 / std::sqrt(BOB0(pos)); }
    [[nodiscard]] auto inv_sqrt_BOB0(CurviCoord const &pos) const noexcept { return std::cos(m_xi * m_D1 * pos.q1); }
    [[nodiscard]] auto BOB0(CurviCoord const &pos) const noexcept { return pow2(1 / inv_sqrt_BOB0(pos)); }

    template <class Coord>
    [[nodiscard]] CartVector impl_covar_basis(Coord const &pos, std::integral_constant<long, 1>) const noexcept { return { m_D1 * BOB0(pos), 0, 0 }; }
    template <class Coord>
    [[nodiscard]] CartVector impl_covar_basis(Coord const &pos, std::integral_constant<long, 2>) const noexcept { return { 0, m_D2 * inv_sqrt_BOB0(pos), 0 }; }
    template <class Coord>
    [[nodiscard]] CartVector impl_covar_basis(Coord const &pos, std::integral_constant<long, 3>) const noexcept { return { 0, 0, m_D3 * inv_sqrt_BOB0(pos) }; }
    template <class Coord>
    [[nodiscard]] CartTensor impl_covar_basis(Coord const &pos, std::integral_constant<long, 0>) const noexcept
    {
        auto const tmp = inv_sqrt_BOB0(pos);
        return { m_D1 / pow2(tmp), m_D2 * tmp, m_D3 * tmp, 0, 0, 0 };
    }

    template <class Coord>
    [[nodiscard]] CartVector impl_contr_basis(Coord const &pos, std::integral_constant<long, 1>) const noexcept { return { m_inv_D1 / BOB0(pos), 0, 0 }; }
    template <class Coord>
    [[nodiscard]] CartVector impl_contr_basis(Coord const &pos, std::integral_constant<long, 2>) const noexcept { return { 0, m_inv_D2 / inv_sqrt_BOB0(pos), 0 }; }
    template <class Coord>
    [[nodiscard]] CartVector impl_contr_basis(Coord const &pos, std::integral_constant<long, 3>) const noexcept { return { 0, 0, m_inv_D3 / inv_sqrt_BOB0(pos) }; }
    template <class Coord>
    [[nodiscard]] CartTensor impl_contr_basis(Coord const &pos, std::integral_constant<long, 0>) const noexcept
    {
        auto const tmp = inv_sqrt_BOB0(pos);
        return { m_inv_D1 * pow2(tmp), m_inv_D2 / tmp, m_inv_D3 / tmp, 0, 0, 0 };
    }

    template <class Coord>
    [[nodiscard]] CovarTensor impl_covar_metric(Coord const &pos) const noexcept
    {
        auto const tmp = pow2(inv_sqrt_BOB0(pos));
        return { pow2(m_D1 / tmp), pow2(m_D2) * tmp, pow2(m_D3) * tmp, 0, 0, 0 };
    }
    template <class Coord>
    [[nodiscard]] ContrTensor impl_contr_metric(Coord const &pos) const noexcept
    {
        auto const tmp = pow2(inv_sqrt_BOB0(pos));
        return { pow2(m_inv_D1 * tmp), pow2(m_inv_D2) / tmp, pow2(m_inv_D3) / tmp, 0, 0, 0 };
    }

public:
    // MARK:- Metric
    /// Calculate covariant components of the metric tensor at a location
    ///
    [[nodiscard]] decltype(auto) covar_metric(CartCoord const &pos) const noexcept { return impl_covar_metric(pos); }
    [[nodiscard]] decltype(auto) covar_metric(CurviCoord const &pos) const noexcept { return impl_covar_metric(pos); }

    /// Calculate contravariant components of the metric tensor at a location
    ///
    [[nodiscard]] decltype(auto) contr_metric(CartCoord const &pos) const noexcept { return impl_contr_metric(pos); }
    [[nodiscard]] decltype(auto) contr_metric(CurviCoord const &pos) const noexcept { return impl_contr_metric(pos); }

    // MARK:- Basis
    /// Calculate i'th covariant basis vectors at a location
    /// \note The index '0' returns all three basis vectors.
    ///
    template <long i>
    [[nodiscard]] decltype(auto) covar_basis(CartCoord const &pos) const noexcept
    {
        static_assert(i >= 0 && i <= 3, "invalid index range");
        return impl_covar_basis(pos, std::integral_constant<long, i>{});
    }
    template <long i>
    [[nodiscard]] decltype(auto) covar_basis(CurviCoord const &pos) const noexcept
    {
        static_assert(i >= 0 && i <= 3, "invalid index range");
        return impl_covar_basis(pos, std::integral_constant<long, i>{});
    }

    /// Calculate i'th contravariant basis vectors at a location
    /// \note The index '0' returns all three basis vectors.
    ///
    template <long i>
    [[nodiscard]] decltype(auto) contr_basis(CartCoord const &pos) const noexcept
    {
        static_assert(i >= 0 && i <= 3, "invalid index range");
        return impl_contr_basis(pos, std::integral_constant<long, i>{});
    }
    template <long i>
    [[nodiscard]] decltype(auto) contr_basis(CurviCoord const &pos) const noexcept
    {
        static_assert(i >= 0 && i <= 3, "invalid index range");
        return impl_contr_basis(pos, std::integral_constant<long, i>{});
    }

    // MARK:- Vector Transformation
    /// Vector transformation from contravariant to covariant components
    /// \param contr Contravariant components of a vector.
    /// \param covar_metric Covariant components of the metric tensor at a given location.
    /// \return Covariant components of the transformed vector.
    [[nodiscard]] static decltype(auto) contr_to_covar(ContrVector const &contr, CovarTensor const &covar_metric) noexcept { return dot(contr, covar_metric); }

    /// Vector transformation from covariant to contravariant components
    /// \param covar Covariant components of a vector.
    /// \param contr_metric Contravariant components of the metric tensor at a given location.
    /// \return Contravariant components of the transformed vector.
    [[nodiscard]] static decltype(auto) covar_to_contr(CovarVector const &covar, ContrTensor const &contr_metric) noexcept { return dot(covar, contr_metric); }

    /// Vector transformation from Cartesian to contravariant components
    /// \param cart Cartesian components of a vector.
    /// \param contr_bases Three contravariant basis vectors at a given location.
    /// \return Contravariant components of the transformed vector.
    [[nodiscard]] static decltype(auto) cart_to_contr(CartVector const &cart, CartTensor const &contr_bases) noexcept { return ContrVector{ dot(contr_bases, cart) }; }

    /// Vector transformation from contravariant to Cartesian components
    /// \param contr Contravariant components of a vector.
    /// \param covar_bases Three covariant basis vectors at a given location.
    /// \return Cartesian components of the transformed vector.
    [[nodiscard]] static decltype(auto) contr_to_cart(ContrVector const &contr, CartTensor const &covar_bases) noexcept { return dot(CartVector{ contr }, covar_bases); }

    /// Vector transformation from Cartesian to covaraint components
    /// \param cart Cartesian components of a vector.
    /// \param covar_bases Three covariant basis vectors at a given location.
    /// \return Covariant components of the transformed vector.
    [[nodiscard]] static decltype(auto) cart_to_covar(CartVector const &cart, CartTensor const &covar_bases) noexcept { return CovarVector{ dot(covar_bases, cart) }; }

    /// Vector transformation from covariant to Cartesian components
    /// \param covar Covariant components of a vector.
    /// \param contr_bases Three contravariant basis vectors at a given location.
    /// \return Cartesian components of the transformed vector.
    [[nodiscard]] static decltype(auto) covar_to_cart(CovarVector const &covar, CartTensor const &contr_bases) noexcept { return dot(CartVector{ covar }, contr_bases); }

    /// Vector transformation from contravariant to covariant components
    /// \tparam Coord Coordinate type.
    /// \param contr Contravariant components of a vector.
    /// \param pos Current location.
    /// \return Covariant components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) contr_to_covar(ContrVector const &contr, Coord const &pos) const noexcept { return contr_to_covar(contr, covar_metric(pos)); }

    /// Vector transformation from covariant to contravariant components
    /// \tparam Coord Coordinate type.
    /// \param covar Covariant components of a vector.
    /// \param pos Current location.
    /// \return Contravariant components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) covar_to_contr(CovarVector const &covar, Coord const &pos) const noexcept { return covar_to_contr(covar, contr_metric(pos)); }

    /// Vector transformation from Cartesian to contravariant components
    /// \tparam Coord Coordinate type.
    /// \param cart Cartesian components of a vector.
    /// \param pos Current location.
    /// \return Contravariant components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) cart_to_contr(CartVector const &cart, Coord const &pos) const noexcept { return cart_to_contr(cart, contr_basis<0>(pos)); }

    /// Vector transformation from contravariant to Cartesian components
    /// \tparam Coord Coordinate type.
    /// \param contr Contravariant components of a vector.
    /// \param pos Current location.
    /// \return Cartesian components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) contr_to_cart(ContrVector const &contr, Coord const &pos) const noexcept { return contr_to_cart(contr, covar_basis<0>(pos)); }

    /// Vector transformation from Cartesian to covaraint components
    /// \tparam Coord Coordinate type.
    /// \param cart Cartesian components of a vector.
    /// \param pos Current location.
    /// \return Covariant components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) cart_to_covar(CartVector const &cart, Coord const &pos) const noexcept { return cart_to_covar(cart, covar_basis<0>(pos)); }

    /// Vector transformation from covariant to Cartesian components
    /// \tparam Coord Coordinate type.
    /// \param covar Covariant components of a vector.
    /// \param pos Current location.
    /// \return Cartesian components of the transformed vector.
    template <class Coord, std::enable_if_t<std::is_same_v<Coord, CartCoord> || std::is_same_v<Coord, CurviCoord>, int> = 0>
    [[nodiscard]] decltype(auto) covar_to_cart(CovarVector const &covar, Coord const &pos) const noexcept { return covar_to_cart(covar, contr_basis<0>(pos)); }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
