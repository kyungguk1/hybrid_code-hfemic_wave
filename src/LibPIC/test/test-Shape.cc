/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Shape.h>
#include <PIC/UTL/println.h>
#include <iostream>

TEST_CASE("Test LibPIC::Shape_1", "[LibPIC::Shape_1]")
{
    constexpr auto is_equal = [](double a, double b) {
        double const eps = 1e-10;
        return std::abs(a - b) < eps;
    };

    {
        Shape<1> sh;
        CHECK(1 == decltype(sh)::order());
        sh.i<0>() = 3;
        CHECK(3 == sh.i(0));
        sh.w<1>() = 3;
        CHECK(3 == sh.w(1));
    }

    for (double x = 0.; x < 3.; x += .1) {
        Shape<1>   sh{ x };
        auto const i = long(std::floor(x));

        bool tf;
        CHECK((tf = i == sh.i<0>() && i + 1 == sh.i<1>()));
        if (!tf) {
            println(std::cout, "x = ", x, ", i(x) = ", i, ", sh.i[0] = ", sh.i<0>(),
                    ", sh.i[1] = ", sh.i<1>());
        }
        CHECK((tf = is_equal(sh.w<0>() + sh.w<1>(), 1.)));
        if (!tf) {
            println(std::cout, "sh.w[0] = ", sh.w<0>(), ", sh.w[1] = ", sh.w<1>());
        }
        CHECK((tf = is_equal(x - double(i), sh.w<1>())));
        if (!tf) {
            println(std::cout, "sh.w[0] = ", sh.w<0>(), ", sh.w[1] = ", sh.w<1>());
        }
    }
}

TEST_CASE("Test LibPIC::Shape_2", "[LibPIC::Shape_2]")
{
    constexpr auto is_equal = [](double a, double b) {
        double const eps = 1e-10;
        return std::abs(a - b) < eps;
    };

    {
        Shape<2> sh;
        CHECK(2 == decltype(sh)::order());
        sh.i<0>() = 3;
        CHECK(3 == sh.i(0));
        sh.w<1>() = 3;
        CHECK(3 == sh.w(1));
    }

    for (double x = 10.; x < 11.; x += .1) {
        Shape<2>   sh{ x };
        auto const i = long(std::round(x));

        bool tf;
        CHECK((tf = i - 1 == sh.i<0>() && i + 0 == sh.i<1>() && i + 1 == sh.i<2>()));
        if (!tf) {
            println(std::cout, "x = ", x, ", i(round(x)) = ", i, ", sh.i[0] = ", sh.i<0>(),
                    ", sh.i[1] = ", sh.i<1>(), ", sh.i[2] = ", sh.i<2>());
        }
        CHECK((tf = is_equal(sh.w<0>() + sh.w<1>() + sh.w<2>(), 1.)));
        if (!tf) {
            println(std::cout, "sh.w[0] = ", sh.w<0>(), ", sh.w[1] = ", sh.w<1>(),
                    ", sh.w[2] = ", sh.w<2>());
        }
    }
}

TEST_CASE("Test LibPIC::Shape_3", "[LibPIC::Shape_3]")
{
    constexpr auto is_equal = [](double a, double b) {
        double const eps = 1e-10;
        return std::abs(a - b) < eps;
    };

    {
        Shape<3> sh;
        CHECK(3 == decltype(sh)::order());
        sh.i<0>() = 3;
        CHECK(3 == sh.i(0));
        sh.w<1>() = 3;
        CHECK(3 == sh.w(1));
    }

    for (double x = 10.; x < 11.; x += .1) {
        Shape<3>   sh{ x };
        auto const i = long(std::ceil(x));

        bool tf;
        CHECK((tf = i - 2 == sh.i<0>() && i - 1 == sh.i<1>() && i + 0 == sh.i<2>()
                 && i + 1 == sh.i<3>()));
        if (!tf) {
            println(std::cout, "x = ", x, ", i(ceil(x)) = ", i, ", sh.i[0] = ", sh.i<0>(),
                    ", sh.i[1] = ", sh.i<1>(), ", sh.i[2] = ", sh.i<2>(),
                    ", sh.i[3] = ", sh.i<3>());
        }
        CHECK((tf = is_equal(sh.w<0>() + sh.w<1>() + sh.w<2>() + sh.w<3>(), 1.)));
        if (!tf) {
            println(std::cout, "sh.w[0] = ", sh.w<0>(), ", sh.w[1] = ", sh.w<1>(),
                    ", sh.w[2] = ", sh.w<2>(), ", sh.w[3] = ", sh.w<3>());
        }
    }
}
