/*
 * Copyright (c) 2022-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Config.h"
#include "Misc/RootFinder.h"
#include "Predefined.h"
#include "UTL/Range.h"
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
namespace {
template <class Function>
[[nodiscard]] Real find_dq1_of_dN(Real const dN, Real const q1, Function const &eta)
{
    static_assert(std::is_invocable_r_v<Real, Function, Real>);

    // dN = eta*dq1
    auto const pred = [dN = std::abs(dN)](auto const iterations, std::pair<Real, Real> const xy, auto) {
        constexpr auto max_iterations = 10'000U;
        constexpr Real rel_tol        = 1e-8;
        if (iterations > max_iterations)
            throw std::domain_error{ std::string{ __PRETTY_FUNCTION__ } + " - no root found after maximum iterations" };
        return std::abs(xy.second) > dN * rel_tol;
    };
    auto const simpson = [&eta](Real const ql, Real const qr) {
        auto const qm = 0.5 * (ql + qr);
        auto const dq = qr - ql;
        Real const dN = (dq / 6) * (eta(ql) + eta(qr) + 4 * eta(qm));
        return dN;
    };
    auto const f = [&](Real const dq) {
        return dN - simpson(q1, q1 + dq);
    };
    Real const dq_guess = dN / eta(q1);
    return secant_while(f, std::make_pair(0, dq_guess), pred).first;
}
template <class Function>
[[nodiscard]] Real integrate_dN(Real const q1_final, Function const &eta)
{
    static_assert(std::is_invocable_r_v<Real, Function, Real>);
    constexpr Real rel_tol = 1e-3;

    // dN = eta*dq1
    Real const dN = std::copysign(rel_tol * eta(0), q1_final);

    // integrate from q1 = 0
    Real q1 = 0;
    Real N  = 0;
    while (std::abs(q1) < std::abs(q1_final)) {
        q1 += find_dq1_of_dN(dN, q1, eta);
        N += dN;
    }
    auto const simpson = [&eta](Real const ql, Real const qr) {
        auto const qm = 0.5 * (ql + qr);
        auto const dq = qr - ql;
        Real const dN = (dq / 6) * (eta(ql) + eta(qr) + 4 * eta(qm));
        return dN;
    };
    return N + simpson(q1, q1_final);
}
template <class Function>
[[nodiscard]] auto build_q1_of_N_interpolation_table(Range const &N_extent, Range const &q1_extent, Function const &eta) -> std::map<Real, Real>
{
    static_assert(std::is_invocable_r_v<Real, Function, Real>);
    constexpr Real rel_tol = 1e-3;

    // dN = eta*dq1
    Real const dN = rel_tol * eta(0);

    Real q1    = q1_extent.min();
    Real N     = N_extent.min();
    auto table = std::map<Real, Real>{ std::make_pair(N, q1) };
    while (N < N_extent.max()) {
        q1 += find_dq1_of_dN(dN, q1, eta);
        N += dN;
        table.emplace_hint(end(table), N, q1);
    }

    // max(q) can exceed domain_extent.max()
    return table;
}

template <class F>
[[nodiscard]] auto init_inverse_function_table(Range const &f_extent, Range const &x_extent, F f_of_x) -> std::map<Real, Real>
{
    static_assert(std::is_invocable_r_v<Real, F, Real>);
    std::map<Real, Real> table;
    table.insert_or_assign(end(table), f_extent.min(), x_extent.min());
    constexpr long n_samples    = 50000;
    constexpr long n_subsamples = 100;
    auto const     df           = f_extent.len / n_samples;
    auto const     dx           = x_extent.len / (n_samples * n_subsamples);
    Real           x            = x_extent.min();
    Real           f_current    = std::invoke(f_of_x, x);
    for (long i = 1; i < n_samples; ++i) {
        Real const f_target = Real(i) * df + f_extent.min();
        while (f_current < f_target)
            f_current = std::invoke(f_of_x, x += dx);
        table.insert_or_assign(end(table), f_current, x);
    }
    table.insert_or_assign(end(table), f_extent.max(), x_extent.max());
    return table;
}
[[nodiscard, maybe_unused]] auto linear_interp(std::map<Real, Real> const &table, Real const x) noexcept -> std::optional<Real>
{
    auto const ub = table.upper_bound(x);
    if (ub == end(table) || ub == begin(table))
        return {};

    auto const &[x_min, y_min] = *std::prev(ub);
    auto const &[x_max, y_max] = *ub;
    return (y_min * (x_max - x) + y_max * (x - x_min)) / (x_max - x_min);
}

// specific to partial shell
template <bool is_small_a>
[[nodiscard]] auto int_cos_zeta(unsigned const zeta, Real const a, Real const x) noexcept -> Real
{
    constexpr auto one_sixth = 0.1666666666666666666666666666666666666667;
    auto const     ax        = a * x;
    if (zeta == 0) {
        return x;
    } else if (zeta == 1) {
        if constexpr (is_small_a) {
            return (1 - ax * ax * one_sixth) * x;
        } else {
            return std::sin(ax) / a;
        }
    } else {
        auto const addendum = int_cos_zeta<is_small_a>(zeta - 2, a, x) * Real(zeta - 1) / zeta;
        if constexpr (is_small_a) {
            return (1 + ax * ax * one_sixth * (2 - 3 * Real(zeta))) * (x / zeta) + addendum;
        } else {
            return std::pow(std::cos(ax), zeta - 1) * std::sin(ax) / (zeta * a) + addendum;
        }
    }
}
} // namespace
LIBPIC_NAMESPACE_END(1)
