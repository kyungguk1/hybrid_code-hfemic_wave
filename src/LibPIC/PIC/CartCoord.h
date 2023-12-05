/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/VT/Vector.h>

#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
struct CartCoord {
    using Real = double;

    Real x{};

    // constructors
    //
    constexpr CartCoord() noexcept = default;
    constexpr explicit CartCoord(Real const x) noexcept
    : x{ x }
    {
    }
    constexpr explicit CartCoord(CartVector const &v) noexcept
    : x{ v.x }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Index.
    /// \param v Vector.
    /// \return Indexed value.
    template <long I>
    [[nodiscard]] constexpr friend auto &get(CartCoord const &coord) noexcept
    {
        static_assert(I >= 0 && I < 1, "index out of range");
        return impl_get<I>(coord);
    }
    template <long I>
    [[nodiscard]] constexpr friend auto &get(CartCoord &coord) noexcept
    {
        static_assert(I >= 0 && I < 1, "index out of range");
        return impl_get<I>(coord);
    }

    // compound operations: coord @= coord, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(CartCoord &lhs, CartCoord const &rhs) noexcept
    {
        lhs.x += rhs.x;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(CartCoord &lhs, CartCoord const &rhs) noexcept
    {
        lhs.x -= rhs.x;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(CartCoord &lhs, CartCoord const &rhs) noexcept
    {
        lhs.x *= rhs.x;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(CartCoord &lhs, CartCoord const &rhs) noexcept
    {
        lhs.x /= rhs.x;
        return lhs;
    }

    // compound operations: coord @= real, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(CartCoord &lhs, Real const &rhs) noexcept
    {
        lhs.x += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(CartCoord &lhs, Real const &rhs) noexcept
    {
        lhs.x -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(CartCoord &lhs, Real const &rhs) noexcept
    {
        lhs.x *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(CartCoord &lhs, Real const &rhs) noexcept
    {
        lhs.x /= rhs;
        return lhs;
    }

    // binary operations: coord @ {coord|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr auto operator+(CartCoord a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator-(CartCoord a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator*(CartCoord a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator/(CartCoord a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ coord, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr auto operator+(Real const &b, CartCoord const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr auto operator-(Real const &a, CartCoord const &b) noexcept
    {
        CartCoord A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr auto operator*(Real const &b, CartCoord const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr auto operator/(Real const &a, CartCoord const &b) noexcept
    {
        CartCoord A{ a };
        A /= b;
        return A;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(CartCoord const &coord) noexcept { return coord; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(CartCoord const &coord) noexcept { return CartCoord{} - coord; }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, CartCoord const &coord)
    {
        return os << '{' << coord.x << '}';
    }

private:
    template <long I, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &coord) noexcept
    {
        if constexpr (I == 0)
            return coord.x;
    }
};

static_assert(std::is_standard_layout_v<CartCoord>);
static_assert(sizeof(CartCoord) / sizeof(double) > 0 && sizeof(CartCoord) % sizeof(double) == 0);
LIBPIC_NAMESPACE_END(1)
