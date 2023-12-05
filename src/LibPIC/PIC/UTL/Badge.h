/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

LIBPIC_NAMESPACE_BEGIN(1)
template <class Holder>
class Badge {
    friend Holder;

    constexpr Badge() noexcept = default;

public:
    Badge(Badge const &) noexcept = delete;
    Badge &operator=(Badge const &) noexcept = delete;
};
LIBPIC_NAMESPACE_END(1)
