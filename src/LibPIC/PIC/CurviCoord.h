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
struct CurviCoord {
    using Real = double;

    Real q1{};

    // constructors
    //
    constexpr CurviCoord() noexcept = default;
    constexpr explicit CurviCoord(Real const q1) noexcept
    : q1{ q1 }
    {
    }
    constexpr explicit CurviCoord(ContrVector const &v) noexcept
    : q1{ v.x }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Index.
    /// \param v Vector.
    /// \return Indexed value.
    template <long I>
    [[nodiscard]] constexpr friend auto &get(CurviCoord const &coord) noexcept
    {
        static_assert(I >= 0 && I < 1, "index out of range");
        return impl_get<I>(coord);
    }
    template <long I>
    [[nodiscard]] constexpr friend auto &get(CurviCoord &coord) noexcept
    {
        static_assert(I >= 0 && I < 1, "index out of range");
        return impl_get<I>(coord);
    }

    // compound operations: coord @= coord, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(CurviCoord &lhs, CurviCoord const &rhs) noexcept
    {
        lhs.q1 += rhs.q1;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(CurviCoord &lhs, CurviCoord const &rhs) noexcept
    {
        lhs.q1 -= rhs.q1;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(CurviCoord &lhs, CurviCoord const &rhs) noexcept
    {
        lhs.q1 *= rhs.q1;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(CurviCoord &lhs, CurviCoord const &rhs) noexcept
    {
        lhs.q1 /= rhs.q1;
        return lhs;
    }

    // compound operations: coord @= real, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(CurviCoord &lhs, Real const &rhs) noexcept
    {
        lhs.q1 += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(CurviCoord &lhs, Real const &rhs) noexcept
    {
        lhs.q1 -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(CurviCoord &lhs, Real const &rhs) noexcept
    {
        lhs.q1 *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(CurviCoord &lhs, Real const &rhs) noexcept
    {
        lhs.q1 /= rhs;
        return lhs;
    }

    // binary operations: coord @ {coord|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr auto operator+(CurviCoord a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator-(CurviCoord a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator*(CurviCoord a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator/(CurviCoord a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ coord, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr auto operator+(Real const &b, CurviCoord const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr auto operator-(Real const &a, CurviCoord const &b) noexcept
    {
        CurviCoord A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr auto operator*(Real const &b, CurviCoord const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr auto operator/(Real const &a, CurviCoord const &b) noexcept
    {
        CurviCoord A{ a };
        A /= b;
        return A;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(CurviCoord const &coord) noexcept { return coord; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(CurviCoord const &coord) noexcept { return CurviCoord{} - coord; }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, CurviCoord const &coord)
    {
        return os << '{' << coord.q1 << '}';
    }

private:
    template <long I, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &coord) noexcept
    {
        if constexpr (I == 0)
            return coord.q1;
    }
};

static_assert(std::is_standard_layout_v<CurviCoord>);
static_assert(sizeof(CurviCoord) / sizeof(double) > 0 && sizeof(CurviCoord) % sizeof(double) == 0);
LIBPIC_NAMESPACE_END(1)
