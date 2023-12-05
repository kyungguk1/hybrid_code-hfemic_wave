/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
struct Scalar {
    using Real = double;

protected:
    Real v{};

public:
    // value access
    //
    constexpr explicit operator Real() const noexcept { return v; }

    [[nodiscard]] constexpr Real const &operator*() const noexcept { return v; }
    [[nodiscard]] constexpr Real const &operator()() const noexcept { return v; }

    // constructors
    //
    constexpr Scalar() noexcept = default;
    constexpr Scalar(Real const v) noexcept
    : v{ v } {}

    // compound operations
    //
    constexpr Scalar &operator+=(Scalar const &o) noexcept
    {
        v += Real{ o };
        return *this;
    }
    constexpr Scalar &operator-=(Scalar const &o) noexcept
    {
        v -= Real{ o };
        return *this;
    }
    constexpr Scalar &operator*=(Scalar const &o) noexcept
    {
        v *= Real{ o };
        return *this;
    }
    constexpr Scalar &operator/=(Scalar const &o) noexcept
    {
        v /= Real{ o };
        return *this;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr Scalar const &operator+(Scalar const &s) noexcept { return s; }
    [[nodiscard]] friend constexpr Scalar        operator-(Scalar const &s) noexcept { return -Real{ s }; }

    // binary operations
    //
    [[nodiscard]] friend constexpr auto operator+(Scalar a, Scalar const &b) noexcept
    {
        a += b;
        return a;
    }
    [[nodiscard]] friend constexpr auto operator-(Scalar a, Scalar const &b) noexcept
    {
        a -= b;
        return a;
    }
    [[nodiscard]] friend constexpr auto operator*(Scalar a, Scalar const &b) noexcept
    {
        a *= b;
        return a;
    }
    [[nodiscard]] friend constexpr auto operator/(Scalar a, Scalar const &b) noexcept
    {
        a /= b;
        return a;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Scalar const &s)
    {
        return os << Real{ s };
    }
};

static_assert(std::is_standard_layout_v<Scalar>);
LIBPIC_NAMESPACE_END(1)
