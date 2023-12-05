/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "ColdSpecies.h"
#include "BField.h"
#include "EField.h"

#include <algorithm>

HYBRID1D_BEGIN_NAMESPACE
ColdSpecies::ColdSpecies(ParamSet const &params, ColdPlasmaDesc const &desc)
: Species{ params }, desc{ desc }
{
}
void ColdSpecies::populate(long, long)
{
    // initialize equilibrium moments
    Real const n0    = 1;
    Real const nV0   = 0;
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < mom1_full.size(); ++i) { // only the interior
        CurviCoord const pos{ i + q1min };
        mom0_full[i] = n0;
        mom1_full[i] = nV0 * geomtr.e1(pos);
    }
}

void ColdSpecies::update_vel(BField const &, EField const &efield, Real const dt)
{
    impl_update_nV(mom1_full, mom0_full, efield, BorisPush{ dt, params.c, params.O0, desc.Oc });
}
void ColdSpecies::impl_update_nV(Grid<CartVector> &nV, Grid<Scalar> const &n, EField const &E, BorisPush const &boris) const
{
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < nV.size(); ++i) {
        CurviCoord const pos{ i + q1min };
        boris.non_relativistic(nV[i], geomtr.Bcart(pos), E[i] * Real{ n[i] });
    }
}

void ColdSpecies::collect_part()
{
    impl_collect_part(moment<0>(), moment<1>());
}
void ColdSpecies::collect_all()
{
    impl_collect_part(moment<0>(), moment<1>());
    impl_collect_nvv(moment<2>(), moment<0>(), moment<1>());
}
void ColdSpecies::impl_collect_part(Grid<Scalar> &n, Grid<CartVector> &nV) const
{
    // must zero-out ghost cells
    n.fill_all(Scalar{});
    nV.fill_all(CartVector{});

    // collect moments
    auto const collect = [w = m_moment_weighting_factor](auto const &mom) {
        return mom * w;
    };
    std::transform(mom0_full.begin(), mom0_full.end(), n.begin(), collect);
    std::transform(mom1_full.begin(), mom1_full.end(), nV.begin(), collect);
}
void ColdSpecies::impl_collect_nvv(Grid<CartTensor> &nvv, Grid<Scalar> const &n, Grid<CartVector> const &nV)
{
    for (long i = 0; i < nV.size(); ++i) {
        CartTensor       &nvvi = nvv[i];
        CartVector const &nVi  = nV[i];
        //
        nvvi.hi() = nvvi.lo()
            = nVi / Real{ n[i] };             // fill diag and off-diag terms with flow velocity
        nvvi.lo() *= nVi;                     // diagonal terms
        nvvi.hi() *= { nVi.y, nVi.z, nVi.x }; // off-diag terms
    }
}
HYBRID1D_END_NAMESPACE
