/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/UTL/Range.h>

TEST_CASE("Test LibPIC::Range", "[LibPIC::Range]")
{
    // properties
    {
        constexpr Range r1{ 0, 0 };
        CHECK(r1.min() == 0);
        CHECK(r1.minmax().first == r1.min());
        CHECK(r1.max() == 0);
        CHECK(r1.minmax().second == r1.max());
        CHECK(r1.mean() == 0);
        CHECK_FALSE(r1.is_member(0));

        constexpr Range r2{ 0, 1 };
        CHECK(r2.min() == 0);
        CHECK(r2.minmax().first == r2.min());
        CHECK(r2.max() == 1);
        CHECK(r2.minmax().second == r2.max());
        CHECK(r2.mean() == .5);
        CHECK(r2.is_member(0));
        CHECK_FALSE(r2.is_member(1));

        constexpr bool tf = std::addressof(r1) == std::addressof(+r1);
        CHECK(tf);

        constexpr Range r3 = -r1;
        CHECK(r3.min() == 0);
        CHECK(r3.max() == 0);
        CHECK(r3.mean() == 0);
        CHECK_FALSE(r3.is_member(0));

        constexpr Range r4 = -r2;
        CHECK(r4.min() == -1);
        CHECK(r4.minmax().first == r4.min());
        CHECK(r4.max() == 0);
        CHECK(r4.minmax().second == r4.max());
        CHECK(r4.mean() == -.5);
        CHECK_FALSE(r4.is_member(0));
        CHECK(r4.is_member(-1));
    }

    // arithematic with scalar
    {
        constexpr auto r1 = Range{ 0, 0 } + 1;
        CHECK(r1.min() == 1);
        CHECK(r1.max() == 1);
        CHECK(r1.mean() == 1);

        constexpr Range r2 = r1 * (-1);
        CHECK(r2.min() == -1);
        CHECK(r2.max() == -1);
        CHECK(r2.mean() == -1);

        constexpr Range r3 = r2 - 1;
        CHECK(r3.min() == -2);
        CHECK(r3.max() == -2);
        CHECK(r3.mean() == -2);

        constexpr Range r4 = r3 / (-2);
        CHECK(r4.min() == 1);
        CHECK(r4.max() == 1);
        CHECK(r4.mean() == 1);
    }
    {
        constexpr auto r1 = 1 + Range{ 0, 0 };
        CHECK(r1.min() == 1);
        CHECK(r1.max() == 1);
        CHECK(r1.mean() == 1);

        constexpr Range r2 = (-1) * r1;
        CHECK(r2.min() == -1);
        CHECK(r2.max() == -1);
        CHECK(r2.mean() == -1);

        constexpr Range r3 = -1 + r2;
        CHECK(r3.min() == -2);
        CHECK(r3.max() == -2);
        CHECK(r3.mean() == -2);
    }
    {
        constexpr auto r1 = Range{ 0, 1 } + 1;
        CHECK(r1.min() == 1);
        CHECK(r1.max() == 2);
        CHECK(r1.mean() == 3. / 2);

        constexpr Range r2 = r1 * (-1);
        CHECK(r2.min() == -2);
        CHECK(r2.max() == -1);
        CHECK(r2.mean() == -3. / 2);

        constexpr Range r3 = r2 - 1;
        CHECK(r3.min() == -3);
        CHECK(r3.max() == -2);
        CHECK(r3.mean() == -5 / 2.);

        constexpr Range r4 = r3 / (-2);
        CHECK(r4.min() == 1);
        CHECK(r4.max() == 1.5);
        CHECK(r4.mean() == 2.5 / 2);
    }
    {
        constexpr auto r1 = 1 + Range{ 0, 1 };
        CHECK(r1.min() == 1);
        CHECK(r1.max() == 2);
        CHECK(r1.mean() == 3. / 2);

        constexpr Range r2 = (-1) * r1;
        CHECK(r2.min() == -2);
        CHECK(r2.max() == -1);
        CHECK(r2.mean() == -3. / 2);

        constexpr Range r3 = -1 + r2;
        CHECK(r3.min() == -3);
        CHECK(r3.max() == -2);
        CHECK(r3.mean() == -5 / 2.);
    }
}
