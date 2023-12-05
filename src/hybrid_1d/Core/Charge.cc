/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Charge.h"
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

Charge::Charge(ParamSet const &params)
: params{ params }
{
}

auto Charge::operator+=(Species const &sp) noexcept -> Charge &
{
    accumulate(this->dead_begin(), sp.moment<0>().dead_begin(), sp.moment<0>().dead_end(), sp.charge_density_conversion_factor());
    return *this;
}
auto Lambda::operator+=(Species const &sp) noexcept -> Lambda &
{
    accumulate(this->dead_begin(), sp.moment<0>().dead_begin(), sp.moment<0>().dead_end(),
               sp.charge_density_conversion_factor() * sp->Oc / params.O0);
    return *this;
}
HYBRID1D_END_NAMESPACE
