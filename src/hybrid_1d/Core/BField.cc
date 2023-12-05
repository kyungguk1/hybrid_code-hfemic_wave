/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "BField.h"
#include "EField.h"

#include <algorithm>

HYBRID1D_BEGIN_NAMESPACE
BField::BField(ParamSet const &params)
: params{ params }, geomtr{ params.geomtr }
{
}

auto BField::operator=(BField const &o) noexcept -> BField &
{ // this is only for the return type casting
    this->GridArray::operator=(o);
    return *this;
}

void BField::update(EField const &efield, Real const dt) noexcept
{
    auto const cdtOsqrtg = dt * params.c / geomtr.sqrt_g();

    // keep the old value and empty current content
    B_prev.operator=(*this);
    this->fill_all(CartVector{});

    // Delta-B followed by phase retardation
    impl_update(*this, cart_to_covar(Ecovar, efield), cdtOsqrtg);
    mask(*this, params.phase_retardation);

    // Next-B followed by amplitude damping
    std::transform(this->begin(), this->end(), B_prev.begin(), this->begin(), std::plus{});
    mask(*this, params.amplitude_damping);
}
void BField::mask(BField &grid, MaskingFunction const &masking_function) const
{
    auto const left_offset  = grid_subdomain_extent().min() - grid_whole_domain_extent().min();
    auto const right_offset = grid_whole_domain_extent().max() - grid_subdomain_extent().max();
    for (long i = 0, first = 0, last = BField::size() - 1; i < BField::size(); ++i) {
        grid[first++] *= masking_function(left_offset + i);
        grid[last--] *= masking_function(right_offset + i);
    }
}

void BField::impl_update(BField &B_cart, Grid<CovarVector> const &E_covar, Real const cdtOsqrtg) const noexcept
{
    auto const curl_E_times_cdt = [cdtOsqrtg](CovarVector const &E1, CovarVector const &E0) noexcept -> ContrVector {
        return {
            0,
            (-E1.z + E0.z) * cdtOsqrtg,
            (+E1.y - E0.y) * cdtOsqrtg,
        };
    };
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < BField::size(); ++i) {
        auto const B_contr = curl_E_times_cdt(E_covar[i - 0], E_covar[i - 1]);
        B_cart[i] -= geomtr.contr_to_cart(B_contr, CurviCoord{ i + q1min });
    }
}
auto BField::cart_to_covar(Grid<CovarVector> &E_covar, EField const &E_cart) -> Grid<CovarVector> &
{
    constexpr auto ghost_offset = 1;
    static_assert(ghost_offset <= Pad);
    auto const q1min = E_cart.grid_subdomain_extent().min();
    for (long i = -ghost_offset; i < EField::size() + ghost_offset; ++i) {
        E_covar[i] = E_cart.geomtr.cart_to_covar(E_cart[i], CurviCoord{ i + q1min });
    }
    return E_covar;
}

namespace {
template <class Object>
decltype(auto) write_attr(Object &obj, [[maybe_unused]] BField const &bfield)
{
    return obj;
}
} // namespace
auto operator<<(hdf5::Group &obj, [[maybe_unused]] BField const &bfield) -> decltype(obj)
{
    return write_attr(obj, bfield);
}
auto operator<<(hdf5::Dataset &obj, [[maybe_unused]] BField const &bfield) -> decltype(obj)
{
    return write_attr(obj, bfield);
}
HYBRID1D_END_NAMESPACE
