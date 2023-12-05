/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include "../PIC/Random/NRRandomEngine.h"
#include <PIC/Predefined.h>
#include <algorithm>
#include <random>
#include <vector>

TEST_CASE("Test LibPIC::NRRandomEngine", "[LibPIC::NRRandomEngine]")
{
    constexpr auto seed = 498547UL;
    constexpr Real min = 1, max = 10;
    auto           rng     = NRRandomEngine{ seed };
    auto           uniform = std::uniform_real_distribution<>{};

    auto const        n_samples = 100000U;
    std::vector<Real> samples(n_samples);
    std::generate(begin(samples), end(samples), [&rng, &uniform]() {
        return uniform(rng) * (max - min) + min;
    });

    std::sort(begin(samples), end(samples));
    CHECK(samples.front() >= min);
    CHECK(samples.back() <= max);

    auto mean_sample = Real{};
    auto var_sample  = Real{};
    for (auto const &sample : samples) {
        mean_sample += sample / n_samples;
        var_sample += sample * sample / n_samples;
    }

    auto const mean_exact = (min + max) / 2;
    CHECK(std::abs(mean_sample - mean_exact) < mean_exact * 1e-2);
    auto const var_exact = (min * min + min * max + max * max) / 3;
    CHECK(std::abs(var_sample - var_exact) < var_exact * 1e-2);

    constexpr auto i = [] {
        auto rng = NRRandomEngine{ 4983U };
        (void)rng();
        (void)rng();
        (void)rng();
        return rng();
    }();
    static_assert(i != 0);
}
