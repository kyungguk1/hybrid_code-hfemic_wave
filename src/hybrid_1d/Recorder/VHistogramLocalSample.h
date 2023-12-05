/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Macros.h"
#include <PIC/Predefined.h>
#include <ParallelKit/ParallelKit.h>

HYBRID1D_BEGIN_NAMESPACE
struct LocalSample {
    unsigned long marker{}; // marker particle count
    Real          weight{}; // weight count
    Real          real_f{}; // full-f count
};
HYBRID1D_END_NAMESPACE

namespace parallel {
template <>
struct TypeMap<H1D::LocalSample> {
    [[nodiscard]] auto operator()() const
    {
        constexpr auto value = H1D::LocalSample{};
        return make_type(value.marker, value.weight, value.real_f);
    }
};
} // namespace parallel
