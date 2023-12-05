/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/UTL/Defer.h>

#include <memory>

TEST_CASE("Test LibPIC::Defer", "[LibPIC::Defer]")
{
    int  count      = 2;
    auto unique_ptr = std::make_unique<long>();
    auto shared_ptr = std::make_shared<long>();

    CHECK(count == 2);
    CHECK(unique_ptr != nullptr);
    CHECK(shared_ptr != nullptr);
    {
        LIBPIC_DEFER(
            [](int &count) {
                --count;
            },
            count);
        LIBPIC_DEFER(
            [&] {
                --count;
            });
        LIBPIC_DEFER(
            [](auto &ptr) {
                ptr = nullptr;
            },
            unique_ptr);
        LIBPIC_DEFER(
            [](auto ptr) {
                ptr = nullptr;
            },
            shared_ptr);
    }
    CHECK(count == 0);
    CHECK(unique_ptr == nullptr);
    CHECK(shared_ptr != nullptr);
}
