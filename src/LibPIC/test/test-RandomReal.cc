/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include "../PIC/RandomReal.h"
#include <algorithm>
#include <vector>

TEST_CASE("Test LibPIC::RandomReal::uniform_real", "[LibPIC::RandomReal::uniform_real]")
{
    constexpr auto seed = 49847UL;
    constexpr Real min = 1, max = 10;

    auto const        n_samples = 100000U;
    std::vector<Real> samples(n_samples);
    std::generate(begin(samples), end(samples), []() {
        return uniform_real<seed>() * (max - min) + min;
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
}

TEST_CASE("Test LibPIC::RandomReal::bit_reversed", "[LibPIC::RandomReal::bit_reversed]")
{
    constexpr auto base = 11U;
    constexpr Real min = 1, max = 10;

    auto const        n_samples = 100000U;
    std::vector<Real> samples(n_samples);
    std::generate(begin(samples), end(samples), []() {
        return bit_reversed<base>() * (max - min) + min;
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
    CHECK(std::abs(mean_sample - mean_exact) < mean_exact * 1e-4);
    auto const var_exact = (min * min + min * max + max * max) / 3;
    CHECK(std::abs(var_sample - var_exact) < var_exact * 1.1e-4);
}
