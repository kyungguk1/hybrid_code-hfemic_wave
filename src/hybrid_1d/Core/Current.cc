/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Current.h"
#include "BField.h"
#include "Charge.h"
#include "EField.h"
#include "Species.h"

HYBRID1D_BEGIN_NAMESPACE
namespace {
template <class LIt, class RIt, class U>
void accumulate(LIt lhs_first, RIt rhs_first, RIt const rhs_last, U const &weight) noexcept
{
    while (rhs_first != rhs_last) {
        *lhs_first++ += *rhs_first++ * weight;
    }
}
} // namespace

Current::Current(ParamSet const &params)
: params{ params }, geomtr{ params.geomtr }
{
}

auto Current::operator+=(Species const &sp) noexcept -> Current &
{
    accumulate(this->dead_begin(), sp.moment<1>().dead_begin(), sp.moment<1>().dead_end(), sp.current_density_conversion_factor());
    return *this;
}
auto Gamma::operator+=(Species const &sp) noexcept -> Gamma &
{
    accumulate(this->dead_begin(), sp.moment<1>().dead_begin(), sp.moment<1>().dead_end(),
               sp.current_density_conversion_factor() * sp->Oc / params.O0);
    return *this;
}

void Current::advance(Lambda const &lambda, Gamma const &gamma, BField const &bfield, EField const &efield, Real const dt) noexcept
{
    impl_advance(*this, lambda, gamma, bfield, efield, dt);
}
void Current::impl_advance(Current &J, Lambda const &L, Gamma const &G, BField const &dB, EField const &E, Real const dt) const noexcept
{
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < Current::size(); ++i) {
        auto const Bi = geomtr.Bcart(CurviCoord{ i + q1min }) + (dB[i + 1] + dB[i + 0]) * 0.5;
        J[i] += (E[i] * Real{ L[i] } + cross(G[i], Bi)) * dt;
    }
}
HYBRID1D_END_NAMESPACE
