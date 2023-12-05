/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Species.h"

HYBRID1D_BEGIN_NAMESPACE
class EField;
class BField;

/// linearized cold fluid
///
class ColdSpecies : public Species {
    ColdPlasmaDesc desc;

public:
    Grid<Scalar>     mom0_full{}; // 0th moment on full grid
    Grid<CartVector> mom1_full{}; // 1st moment on full grid

    [[nodiscard]] ColdPlasmaDesc const *operator->() const noexcept override { return &desc; }

    ColdSpecies &operator=(ColdSpecies const &) = default;
    ColdSpecies(ParamSet const &params, ColdPlasmaDesc const &desc);
    ColdSpecies() = default; // needed for empty std::array
    explicit ColdSpecies(ParamSet const &params)
    : Species{ params } {} // needed for Domain_PC

    /// Load cold species
    /// \note This should only be called by master thread.
    /// \param color This is unused here; just to keep the symmetry with `PartSpecies::populate`.
    /// \param divisor The number of groups to which cold fluid are divided.
    void populate(long color, long divisor);

    // update flow velocity by dt; <v>^n-1/2 -> <v>^n+1/2
    void update_vel(BField const &bfield, EField const &efield, Real dt);

    void collect_part(); // collect 0th & 1st moments
    void collect_all();  // collect all moments

private:
    void impl_update_nV(Grid<CartVector> &nV, Grid<Scalar> const &n, EField const &E, BorisPush const &boris) const;

    void        impl_collect_part(Grid<Scalar> &n, Grid<CartVector> &nV) const;
    static void impl_collect_nvv(Grid<CartTensor> &nvv, Grid<Scalar> const &n, Grid<CartVector> const &nV);
};
HYBRID1D_END_NAMESPACE
