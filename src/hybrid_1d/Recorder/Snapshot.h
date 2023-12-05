/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Core/Domain.h"
#include <PIC/TypeMaps.h>

#include <HDF5Kit/HDF5Kit.h>
#include <ParallelKit/ParallelKit.h>
#include <string>
#include <string_view>

HYBRID1D_BEGIN_NAMESPACE
class Snapshot {
    using interprocess_comm_t = parallel::Communicator<Scalar, CartVector, CartTensor, Particle, long>;
    using rank_t              = parallel::mpi::Rank;

    std::string               filename_suffix{};
    interprocess_comm_t const comm;
    std::size_t const         signature;
    std::string_view const    wd; // working directory

    static constexpr auto   tag  = parallel::mpi::Tag{ 599 };
    static constexpr auto   tag2 = tag + 1;
    static constexpr rank_t master{ 0 };

    [[nodiscard]] bool is_master() const { return master == comm->rank(); }
    [[nodiscard]] auto filepath() const;

public:
    Snapshot(parallel::mpi::Comm subdomain_comm, ParamSet const &params, long subdomain_color);

    // load/save interface
    friend void save(Snapshot &&snapshot, Domain const &domain, long step_count)
    {
        return (snapshot.*snapshot.save)(domain, step_count);
    }
    [[nodiscard]] friend long load(Snapshot &&snapshot, Domain &domain)
    {
        return (snapshot.*snapshot.load)(domain);
    }

private:
    void (Snapshot::*save)(Domain const &domain, long step_count) const &;
    long (Snapshot::*load)(Domain &domain) const &;

    template <class T, long N>
    auto save_helper(hdf5::Group &root, GridArray<T, N, Pad> const &payload, std::string const &basename) const -> hdf5::Dataset;
    void save_helper(hdf5::Group &root, PartSpecies const &payload) const;
    void save_master(Domain const &domain, long step_count) const &;
    void save_worker(Domain const &domain, long step_count) const &;

    template <class T, long N>
    void               load_helper(hdf5::Group const &root, GridArray<T, N, Pad> &payload, std::string const &basename) const;
    void               load_helper(hdf5::Group const &root, PartSpecies &sp) const;
    void               distribute_particles(PartSpecies &sp) const;
    [[nodiscard]] long load_master(Domain &domain) const &;
    [[nodiscard]] long load_worker(Domain &domain) const &;
};
HYBRID1D_END_NAMESPACE
