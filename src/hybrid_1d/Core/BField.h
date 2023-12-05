/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"

#include <HDF5Kit/HDF5Kit.h>

HYBRID1D_BEGIN_NAMESPACE
class EField;

class BField : public Grid<CartVector> {
    Grid<CartVector>  B_prev;
    Grid<CovarVector> Ecovar;

public:
    ParamSet const params;
    Geometry const geomtr;

    explicit BField(ParamSet const &);
    BField &operator=(BField const &o) noexcept;

    [[nodiscard]] auto &grid_whole_domain_extent() const noexcept { return params.full_grid_whole_domain_extent; }
    [[nodiscard]] auto &grid_subdomain_extent() const noexcept { return params.full_grid_subdomain_extent; }

    void update(EField const &efield, Real dt) noexcept;

private:
    void        mask(BField &, MaskingFunction const &masking_function) const;
    void        impl_update(BField &B_cart, Grid<CovarVector> const &Ecovar, Real cdtOsqrtg) const noexcept;
    static auto cart_to_covar(Grid<CovarVector> &Ecovar, EField const &E_cart) -> Grid<CovarVector> &;

    // attribute export facility
    //
    friend auto operator<<(hdf5::Group &obj, BField const &bfield) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, BField const &bfield) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, BField const &bfield) -> decltype(obj)
    {
        return std::move(obj << bfield);
    }
    friend auto operator<<(hdf5::Dataset &&obj, BField const &bfield) -> decltype(obj)
    {
        return std::move(obj << bfield);
    }
};
HYBRID1D_END_NAMESPACE
