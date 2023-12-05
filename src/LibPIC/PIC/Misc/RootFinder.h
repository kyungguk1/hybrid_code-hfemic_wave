/*
 * Copyright (c) 2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Config.h"
#include "../Predefined.h"
#include <type_traits>
#include <utility>

LIBPIC_NAMESPACE_BEGIN(1)
namespace {
/// Root finding using the secant method.
/// \tparam Function A callable with signature `Real(Real)`.
/// \tparam Predicate A predicate function with signature `bool(unsigned, std::pair<Real, Real>, Real)`.
///                   The first argument is the iteration count;
///                   The second argument is a pair of (x, f(x)), where x is the approximate solution;
///                   The third argument is ∆x.
///                   Search stops when it returns false.
/// \param f A function object that returns f(x).
/// \param xinit A pair of initial guesses.
/// \param pred A predicate object.
/// \return A pair of (x, f(x)), where x is the approximate solution.
template <class Function, class Predicate>
[[nodiscard]] auto secant_while(Function const &f, std::pair<Real, Real> const &initial_guesses, Predicate const &pred) -> std::pair<Real, Real>
{
    static_assert(std::is_invocable_r_v<Real, Function, Real>);
    static_assert(std::is_invocable_r_v<bool, Predicate, unsigned, std::pair<Real, Real>, Real>);

    auto x0  = initial_guesses.first;
    auto f0  = f(x0);
    auto x1  = initial_guesses.second;
    auto f1  = f(x1);
    auto idx = 0U;
    while (pred(idx, std::make_pair(x1, f1), x1 - x0)) {
        auto const x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        auto const f2 = f(x2);
        idx += 1;
        x0 = std::exchange(x1, x2);
        f0 = std::exchange(f1, f2);
    }

    return std::make_pair(x1, f1);
}
} // namespace
LIBPIC_NAMESPACE_END(1)
