/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Delegate.h"
#include <PIC/TypeMaps.h>

#include <ParallelKit/ParallelKit.h>

HYBRID1D_BEGIN_NAMESPACE
class SubdomainDelegate : public Delegate {
    using interprocess_comm_t = parallel::Communicator<Scalar, CartVector, CartTensor, Particle>;
    using rank_t              = parallel::mpi::Rank;

    interprocess_comm_t comm;
    rank_t              left_{ -1 };
    rank_t              right{ -1 };

    static constexpr rank_t master{ 0 };
    [[nodiscard]] bool      is_master() const { return master == comm->rank(); }

    // these must be consistent with the definition in ParamSet
    [[nodiscard]] bool is_leftmost_subdomain() const { return comm->rank() == 0; }
    [[nodiscard]] bool is_rightmost_subdomain() const { return comm->rank() == comm.size() - 1; }

public:
    explicit SubdomainDelegate(parallel::mpi::Comm comm);

private:
    void once(Domain &) const override;
    void prologue(Domain const &, long) const override {}
    void epilogue(Domain const &, long) const override {}

    // overrides
    //
    void boundary_pass(PartSpecies const &, BucketBuffer &) const override;
    void boundary_pass(Domain const &, ColdSpecies &) const override;
    void boundary_pass(Domain const &, BField &) const override;
    void boundary_pass(Domain const &, EField &) const override;
    void boundary_pass(Domain const &, Charge &) const override;
    void boundary_pass(Domain const &, Current &) const override;
    void boundary_gather(Domain const &, Charge &) const override;
    void boundary_gather(Domain const &, Current &) const override;
    void boundary_gather(Domain const &, Species &) const override;

    // helpers
    template <class T, long N>
    void mpi_pass(GridArray<T, N, Pad> &) const;

    template <class T, long N>
    void moment_gather(ParamSet const &, GridArray<T, N, Pad> &) const;
    template <class T, long N>
    void mpi_gather(GridArray<T, N, Pad> &) const;

    void mpi_pass(BucketBuffer &) const;
    void periodic_particle_pass(PartSpecies const &, BucketBuffer &) const;
    void reflecting_particle_pass(PartSpecies const &, BucketBuffer &) const;
};
HYBRID1D_END_NAMESPACE
