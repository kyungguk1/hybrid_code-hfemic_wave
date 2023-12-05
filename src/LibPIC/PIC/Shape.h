/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/Predefined.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <ostream>

LIBPIC_NAMESPACE_BEGIN(1)
template <long Order>
struct Shape;

/// 1st-order CIC
///
template <>
struct Shape<1> {
    [[nodiscard]] static constexpr unsigned order() noexcept { return 1; }

    Shape() noexcept = default;
    explicit Shape(Real const x_Dx) noexcept // where x_Dx = x/Dx
    {
        m_i[0] = m_i[1] = long(std::floor(x_Dx));
        m_i[1] += 1;
        m_w[1] = x_Dx - Real(m_i[0]);
        m_w[0] = 1 - m_w[1];
    }

    template <long idx>
    [[nodiscard]] decltype(auto) i() noexcept { return std::get<idx>(m_i); }
    template <long idx>
    [[nodiscard]] decltype(auto) i() const noexcept
    {
        return std::get<idx>(m_i);
    }
    [[nodiscard]] decltype(auto) i(unsigned long idx) const noexcept { return m_i[idx]; }

    template <long idx>
    [[nodiscard]] decltype(auto) w() noexcept { return std::get<idx>(m_w); }
    template <long idx>
    [[nodiscard]] decltype(auto) w() const noexcept
    {
        return std::get<idx>(m_w);
    }
    [[nodiscard]] decltype(auto) w(unsigned long idx) const noexcept { return m_w[idx]; }

private:
    std::array<long, 2> m_i; //!< indices
    std::array<Real, 2> m_w; //!< weights

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Shape const &s)
    {
        return os << "Shape["
                  << "indices = {" << s.m_i[0] << ", " << s.m_i[1] << "}, "
                  << "weights = {" << s.m_w[0] << ", " << s.m_w[1] << '}' << ']';
    }
};

/// 2nd-order TSC
///
template <>
struct Shape<2> {
    [[nodiscard]] static constexpr unsigned order() noexcept { return 2; }

    Shape() noexcept = default;
    explicit Shape(Real x_Dx) noexcept // where x_Dx = x/Dx
    {
        constexpr Real half = 0.5;
        constexpr Real f3_4 = 0.75;

        m_i[0] = m_i[1] = m_i[2] = long(std::round(x_Dx));
        m_i[0] -= 1;
        m_i[2] += 1;
        //
        // i = i1
        //
        x_Dx   = Real(m_i[1]) - x_Dx;  // (i - x)
        m_w[1] = f3_4 - (x_Dx * x_Dx); // i = i1, w1 = 3/4 - (x-i)^2
        //
        // i = i0
        //
        x_Dx += half;                  // (i - x) + 1/2
        m_w[0] = half * (x_Dx * x_Dx); // i = i0, w0 = 1/2 * (1/2 - (x-i))^2
        //
        // i = i2
        //
        m_w[2] = 1 - (m_w[0] + m_w[1]);
    }

    template <long idx>
    [[nodiscard]] decltype(auto) i() noexcept { return std::get<idx>(m_i); }
    template <long idx>
    [[nodiscard]] decltype(auto) i() const noexcept
    {
        return std::get<idx>(m_i);
    }
    [[nodiscard]] decltype(auto) i(unsigned long idx) const noexcept { return m_i[idx]; }

    template <long idx>
    [[nodiscard]] decltype(auto) w() noexcept { return std::get<idx>(m_w); }
    template <long idx>
    [[nodiscard]] decltype(auto) w() const noexcept
    {
        return std::get<idx>(m_w);
    }
    [[nodiscard]] decltype(auto) w(unsigned long idx) const noexcept { return m_w[idx]; }

private:
    std::array<long, 3> m_i; //!< indices
    std::array<Real, 3> m_w; //!< weights

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Shape const &s)
    {
        return os << "Shape["
                  << "indices = {" << s.m_i[0] << ", " << s.m_i[1] << ", " << s.m_i[2] << "}, "
                  << "weights = {" << s.m_w[0] << ", " << s.m_w[1] << ", " << s.m_w[2] << '}'
                  << ']';
    }
};

/// 3rd-order
///
///  W(x) =
///         (2 + x)^3/6           for -2 <= x < -1
///         (4 - 6x^2 - 3x^3)/6   for -1 <= x < 0
///         (4 - 6x^2 + 3x^3)/6   for 0 <= x < 1
///         (2 - x)^3/6           for 1 <= x < 2
///         0                     otherwise
///
template <>
struct Shape<3> {
    [[nodiscard]] static constexpr unsigned order() noexcept { return 3; }

    Shape() noexcept = default;
    explicit Shape(Real const x_Dx) noexcept // where x_Dx = x/Dx
    {
        std::fill(std::begin(m_w), std::end(m_w), 1. / 6);
        m_i[0] = long(std::ceil(x_Dx)) - 2;
        m_i[1] = m_i[0] + 1;
        m_i[2] = m_i[1] + 1;
        m_i[3] = m_i[2] + 1;

        // for -2 <= x < -1
        //
        Real tmp;
        tmp = 2 + (Real(m_i[0]) - x_Dx); // -1 + i0 - x
        m_w[0] *= tmp * tmp * tmp;

        // for 1 <= x < 2
        //
        tmp = 2 - (Real(m_i[3]) - x_Dx); // 2 + i0 - x
        m_w[3] *= tmp * tmp * tmp;

        // for -1 <= x < 0
        //
        tmp = x_Dx - Real(m_i[1]); // x - i0
        m_w[1] *= 4 + 3 * tmp * tmp * (tmp - 2);

        // for 0 <= x < 1
        //
        m_w[2] = 1 - (m_w[0] + m_w[1] + m_w[3]);
    }

    template <long idx>
    [[nodiscard]] decltype(auto) i() noexcept { return std::get<idx>(m_i); }
    template <long idx>
    [[nodiscard]] decltype(auto) i() const noexcept
    {
        return std::get<idx>(m_i);
    }
    [[nodiscard]] decltype(auto) i(unsigned long idx) const noexcept { return m_i[idx]; }

    template <long idx>
    [[nodiscard]] decltype(auto) w() noexcept { return std::get<idx>(m_w); }
    template <long idx>
    [[nodiscard]] decltype(auto) w() const noexcept
    {
        return std::get<idx>(m_w);
    }
    [[nodiscard]] decltype(auto) w(unsigned long idx) const noexcept { return m_w[idx]; }

private:
    std::array<long, 4> m_i; //!< indices
    std::array<Real, 4> m_w; //!< weights

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Shape const &s)
    {
        return os << "Shape["
                  << "indices = {" << s.m_i[0] << ", " << s.m_i[1] << ", " << s.m_i[2] << ", "
                  << s.m_i[3] << "}, "
                  << "weights = {" << s.m_w[0] << ", " << s.m_w[1] << ", " << s.m_w[2] << ", "
                  << s.m_w[3] << '}' << ']';
    }
};
LIBPIC_NAMESPACE_END(1)
