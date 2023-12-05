/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/CartCoord.h>
#include <PIC/CurviCoord.h>

using std::get;

TEST_CASE("Test LibPIC::CartCoord", "[LibPIC::CartCoord]")
{
    constexpr auto coord1 = CartCoord{};
    STATIC_REQUIRE(coord1.x == 0);

    constexpr auto coord5 = CartCoord{ 5 };
    STATIC_REQUIRE(coord5.x == 5);

    constexpr auto coord6 = [](CartCoord coord) {
        get<0>(coord) = 6;
        return coord;
    }({});
    STATIC_REQUIRE(get<0>(coord6) == 6);

    constexpr auto cv     = CartVector{ 1, 2, 3 };
    constexpr auto coord2 = CartCoord{ cv };
    STATIC_REQUIRE(coord2.x == 1);

    constexpr auto coord3 = +coord2;
    STATIC_REQUIRE(coord3.x == coord2.x);

    constexpr auto coord4 = -coord2;
    STATIC_REQUIRE(coord4.x == -coord2.x);

    {
        constexpr auto res1 = coord2 + coord4;
        STATIC_REQUIRE(res1.x == 0);
        constexpr auto res2 = coord2 + coord4.x;
        STATIC_REQUIRE(res2.x == 0);
        constexpr auto res3 = coord2.x + coord4;
        STATIC_REQUIRE(res3.x == 0);
    }
    {
        constexpr auto res1 = coord2 - coord3;
        STATIC_REQUIRE(res1.x == 0);
        constexpr auto res2 = coord2 - coord3.x;
        STATIC_REQUIRE(res2.x == 0);
        constexpr auto res3 = coord2.x - coord3;
        STATIC_REQUIRE(res3.x == 0);
    }
    {
        constexpr auto res1 = coord2 * coord4;
        STATIC_REQUIRE(res1.x == -1);
        constexpr auto res2 = coord2 * coord4.x;
        STATIC_REQUIRE(res2.x == -1);
        constexpr auto res3 = coord2.x * coord4;
        STATIC_REQUIRE(res3.x == -1);
    }
    {
        constexpr auto res1 = coord2 / coord4;
        STATIC_REQUIRE(res1.x == -1);
        constexpr auto res2 = coord2 / coord4.x;
        STATIC_REQUIRE(res2.x == -1);
        constexpr auto res3 = coord2.x / coord4;
        STATIC_REQUIRE(res3.x == -1);
    }
}

TEST_CASE("Test LibPIC::CurviCoord", "[LibPIC::CurviCoord]")
{
    constexpr auto coord1 = CurviCoord{};
    STATIC_REQUIRE(coord1.q1 == 0);

    constexpr auto coord5 = CurviCoord{ 5 };
    STATIC_REQUIRE(coord5.q1 == 5);

    constexpr auto coord6 = [](CurviCoord coord) {
        get<0>(coord) = 6;
        return coord;
    }({});
    STATIC_REQUIRE(get<0>(coord6) == 6);

    constexpr auto cv     = ContrVector{ 1, 2, 3 };
    constexpr auto coord2 = CurviCoord{ cv };
    STATIC_REQUIRE(coord2.q1 == 1);

    constexpr auto coord3 = +coord2;
    STATIC_REQUIRE(coord3.q1 == coord2.q1);

    constexpr auto coord4 = -coord2;
    STATIC_REQUIRE(coord4.q1 == -coord2.q1);

    {
        constexpr auto res1 = coord2 + coord4;
        STATIC_REQUIRE(res1.q1 == 0);
        constexpr auto res2 = coord2 + coord4.q1;
        STATIC_REQUIRE(res2.q1 == 0);
        constexpr auto res3 = coord2.q1 + coord4;
        STATIC_REQUIRE(res3.q1 == 0);
    }
    {
        constexpr auto res1 = coord2 - coord3;
        STATIC_REQUIRE(res1.q1 == 0);
        constexpr auto res2 = coord2 - coord3.q1;
        STATIC_REQUIRE(res2.q1 == 0);
        constexpr auto res3 = coord2.q1 - coord3;
        STATIC_REQUIRE(res3.q1 == 0);
    }
    {
        constexpr auto res1 = coord2 * coord4;
        STATIC_REQUIRE(res1.q1 == -1);
        constexpr auto res2 = coord2 * coord4.q1;
        STATIC_REQUIRE(res2.q1 == -1);
        constexpr auto res3 = coord2.q1 * coord4;
        STATIC_REQUIRE(res3.q1 == -1);
    }
    {
        constexpr auto res1 = coord2 / coord4;
        STATIC_REQUIRE(res1.q1 == -1);
        constexpr auto res2 = coord2 / coord4.q1;
        STATIC_REQUIRE(res2.q1 == -1);
        constexpr auto res3 = coord2.q1 / coord4;
        STATIC_REQUIRE(res3.q1 == -1);
    }
}
