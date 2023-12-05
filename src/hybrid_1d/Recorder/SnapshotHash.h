/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Macros.h"

#include <cstddef> // std::size_t
#include <tuple>
#include <type_traits> // std::hash
#include <utility>

HYBRID1D_BEGIN_NAMESPACE
template <class Tuple>
struct Hash;
template <class T>
Hash(T const &) -> Hash<T>;
//
template <class... Ts>
struct Hash<std::tuple<Ts...>> {
    std::tuple<Ts...> const t;

    [[nodiscard]] constexpr auto operator()() const noexcept
    {
        return hash(std::index_sequence_for<Ts...>{});
    }

private:
    template <std::size_t... Is>
    [[nodiscard]] constexpr std::size_t hash(std::index_sequence<Is...>) const noexcept
    {
        std::size_t hash = 0;
        return (..., ((hash <<= 1) ^= this->hash(std::get<Is>(t))));
    }
    template <class T>
    [[nodiscard]] static constexpr std::size_t hash(T const &x) noexcept
    {
        return std::hash<T>{}(x);
    }
};
HYBRID1D_END_NAMESPACE
