/*
 * Copyright (c) 2017-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Config.h"

#include <limits>

LIBPIC_NAMESPACE_BEGIN(1)
/// Random number generator engine from "Numerical Recipe," Press, 2007, Chapter 7.1.
///
class NRRandomEngine final {
public:
    // UniformRandomBitGenerator requirement
    using result_type = unsigned long;

    [[nodiscard]] static constexpr auto min() noexcept
    {
        return std::numeric_limits<result_type>::min();
    }
    [[nodiscard]] static constexpr auto max() noexcept
    {
        return std::numeric_limits<result_type>::max();
    }

    [[nodiscard]] constexpr result_type operator()() noexcept { return variate_int(); }

    // ctor
    constexpr NRRandomEngine(result_type const seed) noexcept
    {
        m_u = seed ^ m_v;
        (void)variate_int();
        m_v = m_u;
        (void)variate_int();
        m_w = m_v;
        (void)variate_int();
    }

    // disable copy/move
    NRRandomEngine(NRRandomEngine const &) = delete;
    NRRandomEngine &operator=(NRRandomEngine const &) = delete;

private:
    [[nodiscard]] constexpr auto variate_real() noexcept
    {
        return 5.42101086242752217e-20 * variate_int();
    }
    [[nodiscard]] constexpr result_type variate_int() noexcept
    {
        m_u = m_u * 2862933555777941757UL + 7046029254386353087UL;
        m_v ^= m_v >> 17U;
        m_v ^= m_v << 31U;
        m_v ^= m_v >> 8U;
        m_w = 4294957665UL * (m_w & 0xffffffff) + (m_w >> 32U);

        auto tmp = m_u ^ (m_u << 21U);
        tmp ^= tmp >> 35U;
        tmp ^= tmp << 4U;
        return (tmp + m_v) ^ m_w;
    }

    result_type m_u{ 0UL };
    result_type m_v{ 4101842887655102017UL };
    result_type m_w{ 1UL };
};
LIBPIC_NAMESPACE_END(1)
