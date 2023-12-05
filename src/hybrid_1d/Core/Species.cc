/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Species.h"

HYBRID1D_BEGIN_NAMESPACE
Species::Species(ParamSet const &params)
: params{ params }
, geomtr{ params.geomtr }
, m_moment_weighting_factor{ 1 }
{
    // evenly divide up the source contribution among the distributed particle subdomain clones
    m_moment_weighting_factor /= params.number_of_distributed_particle_subdomain_clones;
}

auto Species::operator=(Species const &other) noexcept -> Species &
{
    m_moment_weighting_factor = other.m_moment_weighting_factor;
    {
        std::tie(this->moment<0>(), this->moment<1>())
            = std::tie(other.moment<0>(), other.moment<1>());
    }
    return *this;
}
auto Species::operator=(Species &&other) noexcept -> Species &
{
    m_moment_weighting_factor = other.m_moment_weighting_factor;
    {
        std::tie(this->moment<0>(), this->moment<1>())
            = std::forward_as_tuple(std::move(other.moment<0>()), std::move(other.moment<1>()));
    }
    return *this;
}

namespace {
template <class Object>
decltype(auto) write_attr(Object &obj, Species const &sp)
{
    using hdf5::make_type;
    using hdf5::Space;
    {
        obj.attribute("Oc", make_type(sp->Oc), Space::scalar()).write(sp->Oc);
        obj.attribute("op", make_type(sp->op), Space::scalar()).write(sp->op);
        obj.attribute("number_of_source_smoothings", make_type(sp->number_of_source_smoothings), Space::scalar())
            .write(sp->number_of_source_smoothings);

        obj.attribute("charge_density_conversion_factor", make_type(sp.charge_density_conversion_factor()), Space::scalar())
            .write(sp.charge_density_conversion_factor());

        obj.attribute("current_density_conversion_factor", make_type(sp.current_density_conversion_factor()), Space::scalar())
            .write(sp.current_density_conversion_factor());

        obj.attribute("energy_density_conversion_factor", make_type(sp.energy_density_conversion_factor()), Space::scalar())
            .write(sp.energy_density_conversion_factor());
    }
    return obj;
}
} // namespace
auto operator<<(hdf5::Group &obj, Species const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
auto operator<<(hdf5::Dataset &obj, Species const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
HYBRID1D_END_NAMESPACE
