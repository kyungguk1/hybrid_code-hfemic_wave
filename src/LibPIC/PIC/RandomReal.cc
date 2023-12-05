/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "RandomReal.h"

#include <map>
#include <stdexcept>
#include <string>

LIBPIC_NAMESPACE_BEGIN(1)
RandomReal::RandomReal(unsigned const seed)
{
    thread_local static std::map<unsigned, RandomReal::engine_t> s_random_real_pool{};

    m_engine = &s_random_real_pool.try_emplace(seed, seed).first->second;
}

template <class... Types>
auto BitReversed::bit_reversed_pool(std::variant<Types...> *)
{
    std::map<unsigned, BitReversed::engine_t> pool;
    if (!(... && pool.try_emplace(Types::base(), std::in_place_type_t<Types>{}).second))
        throw std::domain_error{ __PRETTY_FUNCTION__ };
    return pool;
}
BitReversed::BitReversed(unsigned const base)
{
    thread_local static auto s_bit_reversed_pool = bit_reversed_pool(static_cast<BitReversed::engine_t *>(nullptr));

    if (auto it = s_bit_reversed_pool.find(base); it != end(s_bit_reversed_pool))
        m_engine = &it->second;
    else
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - no engine found for base " + std::to_string(base) };
}
LIBPIC_NAMESPACE_END(1)
