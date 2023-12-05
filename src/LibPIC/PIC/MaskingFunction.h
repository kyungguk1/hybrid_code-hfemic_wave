/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
struct MaskingFunction {
    using Real = double;

    long masking_inset{}; // the length of the masking region near the one-side physical boundary
    Real masking_factor{};

    constexpr MaskingFunction() noexcept = default;
    constexpr MaskingFunction(unsigned masking_inset, Real masking_factor)
    : masking_inset{ masking_inset }, masking_factor{ masking_factor }
    {
        if (masking_factor < 0 || masking_factor > 1)
            throw std::invalid_argument{ __PRETTY_FUNCTION__ };
    }

    [[nodiscard]] constexpr Real operator()(Real const offset) const noexcept
    {
        if (0 == masking_inset || abs(offset) > masking_inset)
            return 1;

        auto const tmp = masking_factor * (1 - abs(offset) / masking_inset);
        return (1 - tmp) * (1 + tmp);
    }

private:
    [[nodiscard]] static constexpr Real abs(Real const x) noexcept { return x < 0 ? -x : x; }
};
LIBPIC_NAMESPACE_END(1)
