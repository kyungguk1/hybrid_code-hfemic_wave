/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Species.h"
#include <PIC/Particle.h>
#include <PIC/VDFVariant.h>

#include <deque>
#include <memory>
#include <vector>

HYBRID1D_BEGIN_NAMESPACE
class EField;
class BField;

/// discrete simulation particle species
///
class PartSpecies : public Species {
    KineticPlasmaDesc           desc;
    std::shared_ptr<VDFVariant> vdf;
    Real                        Nc; //!< number of particles per cell at the equator to be used for normalization

public:
    Grid<Scalar>     equilibrium_mom0{};
    Grid<CartVector> equilibrium_mom1{};
    Grid<CartTensor> equilibrium_mom2{};

    using bucket_type = std::deque<Particle>;
    bucket_type bucket{}; //!< particle container

    [[nodiscard]] KineticPlasmaDesc const *operator->() const noexcept override { return &desc; }

    [[nodiscard]] auto &particle_whole_domain_extent() const noexcept { return params.full_grid_whole_domain_extent; }
    [[nodiscard]] auto &particle_subdomain_extent() const noexcept { return params.full_grid_subdomain_extent; }

    PartSpecies &operator=(PartSpecies const &) = default;
    PartSpecies(ParamSet const &params, KineticPlasmaDesc const &desc, std::unique_ptr<VDFVariant> vdf);
    PartSpecies() = default; // needed for empty std::array
    explicit PartSpecies(ParamSet const &params)
    : Species{ params } {} // needed for Domain_PC

    /// Load particles
    /// \note This should only be called by master thread.
    /// \param color This instructs which particles to keep.
    ///              Say, i'th particle is being loaded. It is kept if `color == i % divisor`.
    /// \param divisor The number of groups to which particles are divided.
    void populate(long color, long divisor);

    /// Load particles from a snapshot
    ///
    /// \tparam Predicate A predicate function of signature bool(Particle) that if returns true, instruct to keep the particle passed in.
    /// \param payload A bag of particles to be loaded.
    /// \param pred A predicate function object. The predicate is called with particles that belong in this subdomain.
    template <class Predicate>
    void load_ptls(std::vector<Particle> const &payload, Predicate &&pred)
    {
        for (auto const &particle : payload) {
            if (particle_subdomain_extent().is_member(particle.pos.q1) && pred(particle))
                bucket.push_back(particle);
        }
    }
    // dump particles
    [[nodiscard]] std::vector<Particle> dump_ptls() const { return { begin(bucket), end(bucket) }; }

    void update_vel(BField const &bfield, EField const &efield, Real dt);
    void update_pos(Real dt, Real fraction_of_grid_size_allowed_to_travel);

    void collect_part(); // collect 0th and 1st moments
    void collect_all();  // collect all moments

private:
    void (PartSpecies::*m_update_velocity)(bucket_type &, Grid<CartVector> const &, EField const &, BorisPush const &) const;
    void (PartSpecies::*m_collect_part)(Grid<Scalar> &, Grid<CartVector> &) const;

    [[nodiscard]] bool impl_update_pos(bucket_type &bucket, Real dt, Real travel_scale_factor) const;

    template <long Order>
    void impl_update_velocity(bucket_type &bucket, Grid<CartVector> const &B, EField const &E, BorisPush const &boris) const;
    void impl_update_weight(bucket_type &bucket, Real nu_dt) const;

    template <long Order>
    void impl_collect_part(Grid<Scalar> &n, Grid<CartVector> &nV) const;
    void impl_collect_all(Grid<Scalar> &n, Grid<CartVector> &nV, Grid<CartTensor> &nvv) const;

    // attribute export facility
    //
    template <class Object>
    friend auto write_attr(Object &obj, PartSpecies const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Group &obj, PartSpecies const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, PartSpecies const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, PartSpecies const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
    friend auto operator<<(hdf5::Dataset &&obj, PartSpecies const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
};
HYBRID1D_END_NAMESPACE
