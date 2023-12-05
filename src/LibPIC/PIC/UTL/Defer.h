/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#ifndef LIBPIC_DEFER
#define LIBPIC_DEFER(...)                                         \
    [[maybe_unused]] auto const LIBPIC_UNIQUE_NAME(_LibPIC_defer) \
        = LIBPIC_NAMESPACE(LIBPIC_INLINE_VERSION)::make_defer(__VA_ARGS__)
#endif

LIBPIC_NAMESPACE_BEGIN(1)
template <class Callback, class... Args>
class Defer {
    static_assert(std::is_invocable_v<Callback, Args &&...>);

    Callback            callback;
    std::tuple<Args...> args;

public:
    Defer(Defer const &) = delete;
    Defer &operator=(Defer const &) = delete;

    ~Defer()
    {
        std::apply(callback, std::move(args));
    }
    Defer(Callback callback, Args &&...args) noexcept((std::is_nothrow_move_constructible_v<Callback> && ... && std::is_nothrow_move_constructible_v<Args>))
    : callback{ std::move(callback) }, args{ std::forward<Args>(args)... }
    {
    }
};
template <class Callback, class... Args>
[[nodiscard]] auto make_defer(Callback callback, Args &&...args) noexcept((std::is_nothrow_move_constructible_v<Callback> && ... && std::is_nothrow_move_constructible_v<Args>))
    -> Defer<Callback, Args...>
{
    return { std::move(callback), std::forward<Args>(args)... };
}
LIBPIC_NAMESPACE_END(1)
