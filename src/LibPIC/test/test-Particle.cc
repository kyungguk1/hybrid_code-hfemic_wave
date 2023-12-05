/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Particle.h>
#include <cmath>

TEST_CASE("Test LibPIC::Particle", "[LibPIC::Particle]")
{
    Particle ptl;
    CHECK(ptl.vel.fold(true, [](bool lhs, auto rhs) {
        return lhs && std::isnan(rhs);
    }));
    CHECK(std::isnan(ptl.pos.q1));
    CHECK(std::isnan(ptl.psd.weight));
    CHECK(std::isnan(ptl.psd.real_f));
    CHECK(std::isnan(ptl.psd.marker));
    CHECK(-1 == ptl.id);

    for (long i = 0; i < 100; ++i) {
        ptl = Particle{ { 1, 2, 3 }, CurviCoord{ 4 } };
        (void)ptl;
    }
    ptl = Particle{ { 1, 2, 3 }, CurviCoord{ 4 } };
    CHECK(ptl.vel.x == 1);
    CHECK(ptl.vel.y == 2);
    CHECK(ptl.vel.z == 3);
    CHECK(ptl.pos.q1 == 4);
    CHECK(std::isnan(ptl.psd.weight));
    CHECK(std::isnan(ptl.psd.real_f));
    CHECK(std::isnan(ptl.psd.marker));
    // CHECK(100 == ptl.id);
}
