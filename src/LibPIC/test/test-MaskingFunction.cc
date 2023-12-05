/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/MaskingFunction.h>

TEST_CASE("Test LibPIC::MaskingFunction", "[LibPIC::MaskingFunction]")
{
    CHECK_THROWS_AS(MaskingFunction(0, 2), std::invalid_argument);
    CHECK_THROWS_AS(MaskingFunction(0, -1), std::invalid_argument);

    constexpr MaskingFunction empty_mask;
    static_assert(empty_mask.masking_inset == 0);
    static_assert(empty_mask.masking_factor == 0);
    static_assert(empty_mask(0) == 1);
    CHECK(empty_mask(0) == Approx{ 1 }.epsilon(1e-10));
    CHECK(empty_mask(1) == Approx{ 1 }.epsilon(1e-10));

    constexpr MaskingFunction zero_inset_mask{ 0, 0.5 };
    static_assert(zero_inset_mask.masking_inset == 0);
    static_assert(zero_inset_mask.masking_factor == 0.5);
    CHECK(zero_inset_mask(0) == Approx{ 1 }.epsilon(1e-10));
    CHECK(zero_inset_mask(1) == Approx{ 1 }.epsilon(1e-10));
    CHECK(zero_inset_mask(2) == Approx{ 1 }.epsilon(1e-10));

    constexpr MaskingFunction zero_factor_mask{ 10, 0 };
    static_assert(zero_factor_mask.masking_inset == 10);
    static_assert(zero_factor_mask.masking_factor == 0);
    CHECK(zero_factor_mask(0) == Approx{ 1 }.epsilon(1e-10));
    CHECK(zero_factor_mask(1) == Approx{ 1 }.epsilon(1e-10));
    CHECK(zero_factor_mask(2) == Approx{ 1 }.epsilon(1e-10));

    constexpr MaskingFunction normal_mask{ 10, 0.5 };
    static_assert(normal_mask.masking_inset == 10);
    static_assert(normal_mask.masking_factor == 0.5);
    CHECK(normal_mask(0) == Approx{ 1 - normal_mask.masking_factor * normal_mask.masking_factor }.epsilon(1e-10));
    CHECK(normal_mask(+normal_mask.masking_inset) == Approx{ 1 }.epsilon(1e-10));
    CHECK(normal_mask(-normal_mask.masking_inset) == Approx{ 1 }.epsilon(1e-10));
    CHECK(normal_mask(+normal_mask.masking_inset + 1) == Approx{ 1 }.epsilon(1e-10));
    CHECK(normal_mask(-normal_mask.masking_inset - 1) == Approx{ 1 }.epsilon(1e-10));
    for (long ioffset = -normal_mask.masking_inset; ioffset <= normal_mask.masking_inset; ++ioffset) {
        auto const doffset = double(ioffset);
        auto const exact   = 1 - std::pow(normal_mask.masking_factor * (std::abs(doffset) - normal_mask.masking_inset) / normal_mask.masking_inset, 2);
        REQUIRE(normal_mask(doffset) == Approx{ exact }.epsilon(1e-10));
    }
}
