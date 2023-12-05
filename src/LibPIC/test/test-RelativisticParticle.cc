/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/RelativisticParticle.h>
#include <cmath>

TEST_CASE("Test LibPIC::RelativisticParticle", "[LibPIC::RelativisticParticle]")
{
    using Particle = RelativisticParticle;

    Particle ptl;
    CHECK(ptl.gcgvel.s.fold(true, [](bool lhs, auto rhs) {
        return lhs && std::isnan(rhs);
    }));
    CHECK(std::isnan(*ptl.gcgvel.t));
    CHECK(std::isnan(ptl.pos.q1));
    CHECK(std::isnan(ptl.psd.weight));
    CHECK(std::isnan(ptl.psd.real_f));
    CHECK(std::isnan(ptl.psd.marker));

    CartVector   v = { 1, 2, 3 };
    double const c = 5;
    double const gamma
        = 1 / std::sqrt((1 - std::sqrt(dot(v, v)) / c) * (1 + std::sqrt(dot(v, v) / c)));
    auto const gv = gamma * v;
    ptl           = Particle{ { gamma * c, gv }, CurviCoord{ 4 } };
    CHECK(*ptl.gcgvel.t == gamma * c);
    CHECK(ptl.gcgvel.s.x == gv.x);
    CHECK(ptl.gcgvel.s.y == gv.y);
    CHECK(ptl.gcgvel.s.z == gv.z);
    CHECK(ptl.pos.q1 == 4);
    CHECK(std::isnan(ptl.psd.weight));
    CHECK(std::isnan(ptl.psd.real_f));
    CHECK(std::isnan(ptl.psd.marker));
    auto const beta = ptl.beta();
    CHECK(beta.x == Approx{ v.x / c }.epsilon(1e-15));
    CHECK(beta.y == Approx{ v.y / c }.epsilon(1e-15));
    CHECK(beta.z == Approx{ v.z / c }.epsilon(1e-15));
    auto const vel = ptl.velocity(c);
    CHECK(vel.x == Approx{ v.x }.epsilon(1e-15));
    CHECK(vel.y == Approx{ v.y }.epsilon(1e-15));
    CHECK(vel.z == Approx{ v.z }.epsilon(1e-15));
}
