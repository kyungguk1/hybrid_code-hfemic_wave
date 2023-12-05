/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Config.h"

#include <limits>
#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
/// @brief Bit reversed pattern from Birdsall and Langdon (1985).
/// @discussion The original implementation is found in Kaijun Liu's PIC code.
///
/// The numbers will repeat once the `sequence` variable wraps around.
/// @note It satisfies the UniformRandomBitGenerator requirement.
///
template <unsigned Base>
class BitReversedPattern final {
    [[nodiscard]] static constexpr bool is_prime(unsigned const prime)
    {
        if (prime < 2)
            throw std::domain_error{ __PRETTY_FUNCTION__ };
        unsigned seq = prime;
        while (prime % --seq) {}
        return 1 == seq;
    }
    static_assert(Base > 1 && is_prime(Base), "Base should be a prime number greater than 1");

public: // UniformRandomBitGenerator requirement
    using result_type = unsigned long;

    [[nodiscard]] static constexpr result_type min() noexcept { return 0; }
    [[nodiscard]] static constexpr result_type max() noexcept { return m_max; }

    [[nodiscard]] constexpr result_type operator()() noexcept { return next_pattern(m_seq++); }

    [[nodiscard]] static constexpr auto base() noexcept { return Base; }

    constexpr BitReversedPattern() noexcept        = default;
    BitReversedPattern(BitReversedPattern const &) = delete;
    BitReversedPattern &operator=(BitReversedPattern const &) = delete;

private:
    [[nodiscard]] static constexpr result_type impl_max(unsigned const base) noexcept
    {
        auto const  max           = std::numeric_limits<result_type>::max() / base;
        result_type power_of_base = base;
        while (power_of_base < max) {
            power_of_base *= base;
        }
        // base^n where n is an integer such that x < std::numeric_limits<result_type>::max()
        return power_of_base;
    }

    [[nodiscard]] static constexpr result_type next_pattern(result_type seq) noexcept
    {
        auto power       = max();
        auto bit_pattern = result_type{ 0 };
        while (seq > 0) {
            bit_pattern += (seq % Base) * (power /= Base);
            seq /= Base;
        }
        return bit_pattern;
    }

    result_type                  m_seq{ 1 }; // sequence
    static constexpr result_type m_max = impl_max(Base);
};
LIBPIC_NAMESPACE_END(1)
