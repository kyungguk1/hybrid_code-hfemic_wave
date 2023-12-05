/*
 * Converted to C++ by Kyungguk Min, August 17, 2021
 * Updated by Kyungguk Min, February 3, 2022
 */

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

 See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// original source code:
// https://github.com/svaarala/duktape/blob/master/misc/splitmix64.c

// original documentation:
/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state; otherwise, we
   rather suggest to use a xoroshiro128+ (for moderately parallel
   computations) or xorshift1024* (for massively parallel computations)
   generator. */

#pragma once

#include "../Config.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

LIBPIC_NAMESPACE_BEGIN(1)
class splitmix64 final {
public: // UniformRandomBitGenerator requirement
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
    constexpr splitmix64(result_type const seed)
    : x{ seed }
    {
        if (seed <= 0)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - seed must be a positive integer up to 64-bit" };
    }

    // disable copy/move
    splitmix64(splitmix64 const &) = delete;
    splitmix64 &operator=(splitmix64 const &) = delete;

private:
    result_type x{}; /* The state can be seeded with any value. */

    [[nodiscard]] constexpr result_type next() noexcept
    {
        result_type z = (x += UINT64_C(0x9E3779B97F4A7C15));
        z             = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z             = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return z ^ (z >> 31);
    }
};
LIBPIC_NAMESPACE_END(1)
