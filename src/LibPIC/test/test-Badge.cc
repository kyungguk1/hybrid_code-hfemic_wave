/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/UTL/Badge.h>

TEST_CASE("Test LibPIC::Badge", "[LibPIC::Badge]")
{
    struct S {
        constexpr auto badge1() const noexcept { return Badge<S>{}; }
        // constexpr auto badge2() const noexcept { return Badge<long>{}; }
    };
    [[maybe_unused]] constexpr auto badge = S{}.badge1();
    REQUIRE_NOTHROW(S{}.badge1());
    // REQUIRE_NOTHROW(S{}.badge2());
}
