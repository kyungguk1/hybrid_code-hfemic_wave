/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Random/BitReversedPattern.h"
#include "Random/NRRandomEngine.h"
#include "Random/xoroshiro128.h"
#include <PIC/Config.h>
#include <PIC/Predefined.h>

#include <random>
#include <variant>

LIBPIC_NAMESPACE_BEGIN(1)
/// Object representing a persistent random real engine
///
class RandomReal final {
public:
    using engine_t = std::mt19937;
    // using engine_t = NRRandomEngine;
    // using engine_t = xoroshiro128;

    [[nodiscard]] Real operator()()
    {
        return m_dist(*m_engine);
    }

    explicit RandomReal(unsigned seed);
    RandomReal(RandomReal const &) = delete;
    RandomReal &operator=(RandomReal const &) = delete;

private:
    engine_t                            *m_engine{};
    std::uniform_real_distribution<Real> m_dist{ eps, 1 - eps };
    static constexpr Real                eps = 1e-15;
};

/// Object representing a persistent bit-reversed bit pattern engine
///
class BitReversed final {
public:
    using engine_t = std::variant<
        BitReversedPattern<2U>,
        BitReversedPattern<3U>,
        BitReversedPattern<5U>,
        BitReversedPattern<7U>,
        BitReversedPattern<11U>,
        BitReversedPattern<13U>,
        BitReversedPattern<17U>,
        BitReversedPattern<19U>,
        BitReversedPattern<23U>,
        BitReversedPattern<29U>>;

    [[nodiscard]] Real operator()()
    {
        return std::visit(
            [this](auto &engine) {
                return m_dist(engine);
            },
            *m_engine);
    }

    explicit BitReversed(unsigned base);
    BitReversed(BitReversed const &) = delete;
    BitReversed &operator=(BitReversed const &) = delete;

private:
    engine_t                            *m_engine{};
    std::uniform_real_distribution<Real> m_dist{ eps, 1 - eps };
    static constexpr Real                eps = 1e-15;

    template <class... Types>
    auto bit_reversed_pool(std::variant<Types...> *);
};

/// Returns a real number (0, 1) following a uniform distribution
/// \tparam seed A seed for a random number generator.
///
template <unsigned seed>
[[nodiscard]] Real uniform_real()
{
    thread_local static auto rng = RandomReal{ seed };
    return rng();
}

/// Returns a real number (0, 1) following a uniform distribution
/// \tparam base Base prime number for BitReversedPattern.
///
template <unsigned base>
[[nodiscard]] Real bit_reversed()
{
    thread_local static auto rng = BitReversed{ base };
    return rng();
}
LIBPIC_NAMESPACE_END(1)
