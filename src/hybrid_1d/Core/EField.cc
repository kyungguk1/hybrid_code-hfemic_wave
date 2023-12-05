/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "EField.h"
#include "BField.h"
#include "Charge.h"
#include "Current.h"

#include <cmath>

HYBRID1D_BEGIN_NAMESPACE
EField::EField(ParamSet const &params)
: params{ params }, geomtr{ params.geomtr }
{
}

auto EField::cart_to_covar(Grid<CovarVector> &B_covar, BField const &B_cart) -> Grid<CovarVector> &
{
    constexpr auto ghost_offset = 1;
    static_assert(ghost_offset <= Pad);
    auto const q1min = B_cart.grid_subdomain_extent().min();
    for (long i = -ghost_offset; i < BField::size() + ghost_offset; ++i) {
        B_covar[i] = B_cart.geomtr.cart_to_covar(B_cart[i], CurviCoord{ i + q1min });
    }
    return B_covar;
}
auto EField::cart_to_contr(Grid<ContrVector> &B_contr, BField const &B_cart) -> Grid<ContrVector> &
{
    constexpr auto ghost_offset = 1;
    static_assert(ghost_offset <= Pad);
    auto const q1min = B_cart.grid_subdomain_extent().min();
    for (long i = -ghost_offset; i < BField::size() + ghost_offset; ++i) {
        B_contr[i] = B_cart.geomtr.cart_to_contr(B_cart[i], CurviCoord{ i + q1min });
    }
    return B_contr;
}

void EField::update(BField const &bfield, Charge const &charge, Current const &current) noexcept
{
    impl_update_dPe(dPe, charge);
    mask(dPe, params.phase_retardation);

    impl_update_Je(Je, current, cart_to_covar(buffer.covar, bfield));
    mask(Je, params.phase_retardation);

    impl_update_E(*this, charge, cart_to_contr(buffer.contr, bfield));
    mask(*this, params.amplitude_damping);
}
template <class T, long N, long Pad>
void EField::mask(GridArray<T, N, Pad> &grid, MaskingFunction const &masking_function) const
{
    auto const left_offset  = grid_subdomain_extent().min() - grid_whole_domain_extent().min();
    auto const right_offset = grid_whole_domain_extent().max() - grid_subdomain_extent().max();
    for (long i = 0, first = 0, last = EField::size() - 1; i < EField::size(); ++i) {
        grid[first++] *= masking_function(left_offset + i);
        grid[last--] *= masking_function(right_offset + i);
    }
}

void EField::impl_update_dPe(Grid<CovarVector> &grad_cPe_covar, Charge const &rho) const noexcept
{
    eFluidDesc const &ef = params.efluid_desc;
    auto             &Pe = buffer.Pe;

    // pressure
    Real const O0        = params.O0;
    Real const O02beO2   = (O0 * O0) * ef.beta * 0.5;
    Real const mOeOO0oe2 = -ef.Oc / (O0 * (ef.op * ef.op));
    for (long i = -Pad; i < EField::size() + Pad; ++i) {
        Pe[i] = std::pow(mOeOO0oe2 * Real{ rho[i] }, ef.gamma) * O02beO2;
    }

    // grad Pe
    auto const grad_Pe_times_c = [cO2 = params.c / 2](Scalar const &Pe_ahead, Scalar const &Pe_behind) noexcept -> CovarVector {
        return {
            Real{ Pe_ahead - Pe_behind } * cO2,
            0,
            0,
        };
    };
    for (long i = 0; i < EField::size(); ++i) {
        grad_cPe_covar[i] = grad_Pe_times_c(Pe[i + 1], Pe[i - 1]);
    }
}
void EField::impl_update_Je(Grid<ContrVector> &Je_contr, Current const &Ji_cart, Grid<CovarVector> const &B_covar) const noexcept
{
    auto const curl_B_times_c = [cOsqrtg = params.c / geomtr.sqrt_g()](CovarVector const &B1, CovarVector const &B0) noexcept -> ContrVector {
        return {
            0,
            (-B1.z + B0.z) * cOsqrtg,
            (+B1.y - B0.y) * cOsqrtg,
        };
    };
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < EField::size(); ++i) {
        Je_contr[i] = curl_B_times_c(B_covar[i + 1], B_covar[i + 0])
                    - geomtr.cart_to_contr(Ji_cart[i], CurviCoord{ i + q1min });
    }
}
void EField::impl_update_E(EField &E_cart, Charge const &rho, Grid<ContrVector> const &dB_contr) const noexcept
{
    Grid<CovarVector> const &grad_cPe_covar = dPe;
    Grid<ContrVector> const &Je_contr       = Je;
    auto const               q1min          = grid_subdomain_extent().min();
    for (long i = 0; i < EField::size(); ++i) {
        CurviCoord const pos{ i + q1min };

        // 1. Je x B term
        //
        auto const B_contr = geomtr.Bcontr(pos) + (dB_contr[i + 1] + dB_contr[i + 0]) * 0.5;
        auto const lorentz = cross(Je_contr[i], B_contr, geomtr.sqrt_g());

        // 2. pressure gradient term
        //
        auto const &grad_cPe = grad_cPe_covar[i];

        // 3. electric field
        //
        auto const E_covar = (lorentz - grad_cPe) / Real{ rho[i] };

        E_cart[i] = geomtr.covar_to_cart(E_covar, pos);
    }
}

namespace {
template <class Object>
decltype(auto) write_attr(Object &obj, [[maybe_unused]] EField const &efield)
{
    return obj;
}
} // namespace
auto operator<<(hdf5::Group &obj, [[maybe_unused]] EField const &efield) -> decltype(obj)
{
    return write_attr(obj, efield);
}
auto operator<<(hdf5::Dataset &obj, [[maybe_unused]] EField const &efield) -> decltype(obj)
{
    return write_attr(obj, efield);
}
HYBRID1D_END_NAMESPACE
