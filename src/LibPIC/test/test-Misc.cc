/*
 * Copyright (c) 2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include "test-Misc.h"
#include <PIC/Misc/Faddeeva.hh>
#include <PIC/Misc/RootFinder.h>

TEST_CASE("Test LibPIC::Math::Dawson", "[LibPIC::Math::Dawson]")
{
    long i = 0;
    for (auto const &[x, f1] : dawson_F) {
        auto const f2 = Faddeeva::Dawson(x);
        INFO("i = " << i << ", x = " << x << ", f1 = " << f1 << ", f2 = " << f2);
        REQUIRE(f1 == Approx{ f2 }.epsilon(1e-10));
        ++i;
    }
}

TEST_CASE("Test LibPIC::Math::RootFinder::Secant", "[LibPIC::Math::RootFinder::Secant]")
{
    auto const x = 20.255019392306682;
    auto const f = [F0 = 0.024715434379990325](Real x) noexcept {
        return Faddeeva::Dawson(x) - F0;
    };
    auto const sol = secant_while(f, std::make_pair(x - 1, x + 1.1), [](auto, auto, auto dx) -> bool {
        return std::abs(dx) > 1e-7;
    });
    CHECK(sol.first == Approx{ x }.epsilon(1e-7));
}
