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
class SnapshotGrid {
    using interprocess_comm_t = parallel::Communicator<Scalar, CartVector, CartTensor, long>;
    using rank_t              = parallel::mpi::Rank;

    interprocess_comm_t const comm;
    std::size_t const         signature;
    std::string_view const    wd; // working directory

    static constexpr auto   tag  = parallel::mpi::Tag{ 599 };
    static constexpr auto   tag2 = tag + 1;
    static constexpr rank_t master{ 0 };

    [[nodiscard]] bool is_master() const { return master == comm->rank(); }
    [[nodiscard]] auto filepath() const;

public:
    SnapshotGrid(parallel::mpi::Comm subdomain_comm, ParamSet const &params);

    // load/save interface
    friend void save(SnapshotGrid &&snapshot, parallel::mpi::Comm const &distributed_particle_comm, Domain const &domain, long step_count)
    {
        if (0 == distributed_particle_comm.rank()) // only master of distributed particle comm group
            (snapshot.*snapshot.save)(domain, step_count);
    }
    [[nodiscard]] friend long load(SnapshotGrid &&snapshot, parallel::mpi::Comm const &distributed_particle_comm, Domain &domain)
    {
        int const  rank   = distributed_particle_comm.rank();
        int const  size   = distributed_particle_comm.size();
        auto       buffer = std::array<char, 1>{};
        auto const type   = parallel::make_type(buffer.front());

        // load one by one in distributed particle comm group; this is to prevent chocking in comm.scatter
        if (0 != rank)
            distributed_particle_comm.recv(buffer.data(), type, buffer.size(), { parallel::mpi::Rank{ rank - 1 }, tag });
        auto const iteration_count = (snapshot.*snapshot.load)(domain);
        if (rank + 1 != size)
            distributed_particle_comm.issend(buffer.data(), type, buffer.size(), { parallel::mpi::Rank{ rank + 1 }, tag }).wait();

        return iteration_count;
    }

private:
    void (SnapshotGrid::*save)(Domain const &domain, long step_count) const &;
    long (SnapshotGrid::*load)(Domain &domain) const &;

    template <class T, long N>
    auto save_helper(hdf5::Group &root, GridArray<T, N, Pad> const &payload, std::string const &basename) const -> hdf5::Dataset;
    void save_master(Domain const &domain, long step_count) const &;
    void save_worker(Domain const &domain, long step_count) const &;

    template <class T, long N>
    void               load_helper(hdf5::Group const &root, GridArray<T, N, Pad> &payload, std::string const &basename) const;
    [[nodiscard]] long load_master(Domain &domain) const &;
    [[nodiscard]] long load_worker(Domain &domain) const &;
};
HYBRID1D_END_NAMESPACE
