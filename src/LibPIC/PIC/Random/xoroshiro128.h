/*
 * Modified by Kyungguk Min, August 17, 2021
 * Updated by Kyungguk Min, February 3, 2022
 *
 * Converted to C++
 * Conformance to the standard's UniformRandomBitGenerator requirement
 */

/* Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// original source code:
// https://prng.di.unimi.it/xoroshiro128plusplus.c

// original documentation:
/* This is xoroshiro128++ 1.0, one of our all-purpose, rock-solid,
   small-state generators. It is extremely (sub-ns) fast and it passes all
   tests we are aware of, but its state space is large enough only for
   mild parallelism.

   For generating just floating-point numbers, xoroshiro128+ is even
   faster (but it has a very mild bias, see notes in the comments).

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

#pragma once

#include "splitmix64.h"

LIBPIC_NAMESPACE_BEGIN(1)
class xoroshiro128 final {
public:
    // UniformRandomBitGenerator requirement
    using result_type = std::uint64_t;

    [[nodiscard]] static constexpr auto min() noexcept
    {
        return std::numeric_limits<result_type>::min();
    }
    [[nodiscard]] static constexpr auto max() noexcept
    {
        return std::numeric_limits<result_type>::max();
    }

    [[nodiscard]] constexpr result_type operator()() noexcept { return next(); }

    // ctor
    constexpr xoroshiro128(result_type const seed)
    {
        if (seed <= 0)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - seed must be a positive integer up to 64-bit" };

        // states initialized by numbers from splitmix64
        auto rng = splitmix64{ seed };
        s0       = rng();
        s1       = rng();
    }

    // disable copy/move
    xoroshiro128(xoroshiro128 const &) = delete;
    xoroshiro128 &operator=(xoroshiro128 const &) = delete;

private:
    result_type s0{};
    result_type s1{};

    // std::rotl is introduced in C++20
    template <int k>
    [[nodiscard]] static constexpr result_type rotl(result_type const x) noexcept
    {
        // this check is necessary since dk will be zero, leading to undefined behavior
        if constexpr (k == 0)
            return x;

        static_assert(k >= 0);
        static_assert(std::numeric_limits<result_type>::radix == 2);
        constexpr auto n_bits = std::numeric_limits<result_type>::digits;
        constexpr auto uk     = result_type{ k % n_bits };
        constexpr auto dk     = result_type{ n_bits - uk };
        return (x << uk) | (x >> dk);
    }

    [[nodiscard]] constexpr result_type next() noexcept
    {
        auto const result = rotl<17>(s0 + s1) + s0;
        {
            s1 ^= s0;
            s0 = rotl<49>(s0) ^ s1 ^ (s1 << 21U); // a, b
            s1 = rotl<28>(s1);                    // c
        }
        return result;
    }
};
LIBPIC_NAMESPACE_END(1)
