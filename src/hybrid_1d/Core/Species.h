/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"
#include <PIC/BorisPush.h>
#include <PIC/UTL/Badge.h>

#include <HDF5Kit/HDF5Kit.h>
#include <tuple>

HYBRID1D_BEGIN_NAMESPACE
class MasterDelegate;
class WorkerDelegate;

/// base class for ion species
///
class Species {
public:
    ParamSet const params;
    Geometry const geomtr;

protected:
    Real m_moment_weighting_factor; //!< This is the scaling factor applied to moment calculation.
public:
    [[nodiscard]] auto &moment_weighting_factor(Badge<MasterDelegate>) &noexcept { return m_moment_weighting_factor; }
    [[nodiscard]] auto &moment_weighting_factor(Badge<WorkerDelegate>) &noexcept { return m_moment_weighting_factor; }

private:
    using MomTuple = std::tuple<Grid<Scalar>, Grid<CartVector>, Grid<CartTensor>>;
    MomTuple m_mom{}; //!< velocity moments at grid points

public:
    // accessors
    //
    [[nodiscard]] virtual PlasmaDesc const *operator->() const noexcept = 0;

    [[nodiscard]] virtual Real charge_density_conversion_factor() const noexcept
    {
        return ((*this)->op * (*this)->op) * params.O0 / (*this)->Oc;
    }
    [[nodiscard]] virtual Real current_density_conversion_factor() const noexcept
    {
        return charge_density_conversion_factor() / params.c;
    }
    [[nodiscard]] virtual Real energy_density_conversion_factor() const noexcept
    {
        Real const tmp = params.O0 / (*this)->Oc * (*this)->op / params.c;
        return tmp * tmp;
    }

    [[nodiscard]] auto &grid_whole_domain_extent() const noexcept { return params.half_grid_whole_domain_extent; }
    [[nodiscard]] auto &grid_subdomain_extent() const noexcept { return params.half_grid_subdomain_extent; }

    // access to i'th velocity moment
    //
    template <long i>
    [[nodiscard]] auto const &moment() const noexcept
    {
        return std::get<i>(m_mom);
    }
    template <long i>
    [[nodiscard]] auto &moment() noexcept { return std::get<i>(m_mom); }
    //
    template <class T>
    [[nodiscard]] auto const &moment() const noexcept
    {
        return std::get<Grid<T>>(m_mom);
    }
    template <class T>
    [[nodiscard]] auto &moment() noexcept { return std::get<Grid<T>>(m_mom); }
    //
    [[nodiscard]] MomTuple const &moments() const noexcept { return m_mom; }
    [[nodiscard]] MomTuple       &moments() noexcept { return m_mom; }

protected:
    virtual ~Species()       = default;
    Species(Species const &) = delete;
    Species(ParamSet const & = {});
    Species &operator=(Species const &) noexcept;
    Species &operator=(Species &&) noexcept;

    // attribute export facility
    //
    friend auto operator<<(hdf5::Group &obj, Species const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, Species const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, Species const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
    friend auto operator<<(hdf5::Dataset &&obj, Species const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
};
HYBRID1D_END_NAMESPACE
