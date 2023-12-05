/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Boundary/Delegate.h"
#include "Domain.h"

HYBRID1D_BEGIN_NAMESPACE
namespace {
template <class T, long N>
auto &operator+=(GridArray<T, N, Pad> &lhs, GridArray<T, N, Pad> const &rhs) noexcept
{
    auto rhs_first = rhs.dead_begin(), rhs_last = rhs.dead_end();
    auto lhs_first = lhs.dead_begin();
    while (rhs_first != rhs_last) {
        *lhs_first++ += *rhs_first++;
    }
    return lhs;
}
//
template <class T, long N>
decltype(auto) operator*=(GridArray<T, N, Pad> &lhs, T const &rhs) noexcept
{ // include padding
    lhs.for_all([&rhs](T &value_ref) {
        value_ref *= rhs;
    });
    return lhs;
}
} // namespace

template <class Species>
auto Domain::collect_smooth(Charge &rho, Species const &sp) const -> Charge const &
{
    rho.reset();
    //
    // collect & gather rho
    //
    rho += sp;
    delegate->boundary_gather(*this, rho);
    //
    // optional smoothing
    //
    for (long i = 0; i < sp->number_of_source_smoothings; ++i) {
        delegate->boundary_pass(*this, rho);
        rho.smooth();
    }
    //
    delegate->boundary_pass(*this, rho);
    return rho;
}
template <class Species>
auto Domain::collect_smooth(Current &J, Species const &sp) const -> Current const &
{
    J.reset();
    //
    // collect & gather J
    //
    J += sp;
    delegate->boundary_gather(*this, J);
    //
    // optional smoothing
    //
    for (long i = 0; i < sp->number_of_source_smoothings; ++i) {
        delegate->boundary_pass(*this, J);
        J.smooth();
    }
    //
    delegate->boundary_pass(*this, J);
    return J;
}
HYBRID1D_END_NAMESPACE
