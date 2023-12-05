/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"

#include <HDF5Kit/HDF5Kit.h>

HYBRID1D_BEGIN_NAMESPACE
class BField;
class Charge;
class Current;

class EField : public Grid<CartVector> {
    struct {
        Grid<CovarVector>    covar;
        Grid<ContrVector>    contr;
        mutable Grid<Scalar> Pe;
    } buffer;
    Grid<ContrVector> Je;
    Grid<CovarVector> dPe; // grad(Pe * c)
    //

public:
    ParamSet const params;
    Geometry const geomtr;

    explicit EField(ParamSet const &);
    EField &operator=(EField const &) = delete;

    [[nodiscard]] auto &grid_whole_domain_extent() const noexcept { return params.half_grid_whole_domain_extent; }
    [[nodiscard]] auto &grid_subdomain_extent() const noexcept { return params.half_grid_subdomain_extent; }

    void update(BField const &bfield, Charge const &charge, Current const &current) noexcept;

private:
    template <class T, long N, long Pad>
    void mask(GridArray<T, N, Pad> &, MaskingFunction const &) const;
    void impl_update_dPe(Grid<CovarVector> &grad_cPe_covar, Charge const &rho) const noexcept;
    void impl_update_Je(Grid<ContrVector> &Je_contr, Current const &Ji_cart, Grid<CovarVector> const &B_covar) const noexcept;
    void impl_update_E(EField &E_cart, Charge const &rho, Grid<ContrVector> const &dB_contr) const noexcept;

    static auto cart_to_covar(Grid<CovarVector> &Bcovar, BField const &B_cart) -> Grid<CovarVector> &;
    static auto cart_to_contr(Grid<ContrVector> &Bcontr, BField const &B_cart) -> Grid<ContrVector> &;

    // attribute export facility
    //
    friend auto operator<<(hdf5::Group &obj, EField const &efield) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, EField const &efield) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, EField const &efield) -> decltype(obj)
    {
        return std::move(obj << efield);
    }
    friend auto operator<<(hdf5::Dataset &&obj, EField const &efield) -> decltype(obj)
    {
        return std::move(obj << efield);
    }
};
HYBRID1D_END_NAMESPACE
