/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "WorkerDelegate.h"
#include "MasterDelegate.h"

HYBRID1D_BEGIN_NAMESPACE
void WorkerDelegate::setup(Domain &domain) const
{
    // distribute particles to workers
    //
    for (PartSpecies &sp : domain.part_species) {
        // receive moment weighting factor from master
        recv_from_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));

        // receive particles from master
        recv_from_master(sp, sp.bucket);
    }

    // distribute cold species to workers
    //
    for (ColdSpecies &sp : domain.cold_species) {
        // receive moment weighting factor from master
        recv_from_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));
    }

    // distribute external sources to workers
    for (ExternalSource &sp : domain.external_sources) {
        // receive moment weighting factor from master
        recv_from_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));
    }
}
template <class Container>
void WorkerDelegate::recv_from_master(PartSpecies const &, Container &bucket) const
{
    // distribute particles to workers
    //
    bucket = comm.recv<Container>(master->comm.rank);
}

void WorkerDelegate::teardown(Domain &domain) const
{
    // collect particles to master
    //
    for (PartSpecies &sp : domain.part_species) {
        // collect moment weighting factor to master
        reduce_to_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));

        // collect particles to master
        collect_to_master(sp, sp.bucket);
    }

    // collect cold species from workers
    //
    for (ColdSpecies &sp : domain.cold_species) {
        // collect moment weighting factor to master
        reduce_to_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));
    }

    // collect external sources to master
    //
    for (ExternalSource &sp : domain.external_sources) {
        // collect moment weighting factor to master
        reduce_to_master(sp.moment_weighting_factor(Badge<WorkerDelegate>{}));
    }
}
template <class Container>
void WorkerDelegate::collect_to_master(PartSpecies const &, Container &bucket) const
{
    // collect particles to master
    //
    comm.send(std::move(bucket), master->comm.rank).wait();
}

void WorkerDelegate::prologue(Domain const &domain, long const i) const
{
    master->delegate->prologue(domain, i);
}
void WorkerDelegate::epilogue(Domain const &domain, long const i) const
{
    master->delegate->epilogue(domain, i);
}
void WorkerDelegate::once(Domain &domain) const
{
    // receive particles' equilibrium moments from master
    //
    for (PartSpecies &sp : domain.part_species) {
        recv_from_master(sp.equilibrium_mom0);
        recv_from_master(sp.equilibrium_mom1);
        recv_from_master(sp.equilibrium_mom2);
    }

    // receive cold species' moments from master
    //
    for (ColdSpecies &sp : domain.cold_species) {
        recv_from_master(sp.mom0_full);
        recv_from_master(sp.mom1_full);
    }

    // receive external sources' current time step from master
    for (ExternalSource &sp : domain.external_sources) {
        sp.set_cur_step(recv_from_master(sp.cur_step()));
    }

    master->delegate->once(domain);
}
void WorkerDelegate::boundary_pass(Domain const &, PartSpecies &sp) const
{
    // be careful not to access it from multiple threads
    // note that the content is cleared after this call
    auto &buffer = bucket_buffer();
    master->delegate->partition(sp, buffer);
    //
    {
        collect_to_master(sp, buffer.L);
        collect_to_master(sp, buffer.R);
    }
    {
        recv_from_master(sp, buffer.L);
        recv_from_master(sp, buffer.R);
    }
    //
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.L), cend(buffer.L));
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.R), cend(buffer.R));
}
void WorkerDelegate::boundary_pass(Domain const &, ColdSpecies &sp) const
{
    recv_from_master(sp.mom0_full);
    recv_from_master(sp.mom1_full);
}
void WorkerDelegate::boundary_pass(Domain const &, BField &bfield) const
{
    recv_from_master(bfield);
}
void WorkerDelegate::boundary_pass(Domain const &, EField &efield) const
{
    recv_from_master(efield);
}
void WorkerDelegate::boundary_pass(Domain const &, Charge &charge) const
{
    recv_from_master(charge);
}
void WorkerDelegate::boundary_pass(Domain const &, Current &current) const
{
    recv_from_master(current);
}
void WorkerDelegate::boundary_gather(Domain const &, Charge &charge) const
{
    reduce_to_master(charge);
    recv_from_master(charge);
}
void WorkerDelegate::boundary_gather(Domain const &, Current &current) const
{
    reduce_to_master(current);
    recv_from_master(current);
}
void WorkerDelegate::boundary_gather(Domain const &, Species &sp) const
{
    {
        reduce_to_master(sp.moment<0>());
        reduce_to_master(sp.moment<1>());
        reduce_to_master(sp.moment<2>());
    }
    {
        recv_from_master(sp.moment<0>());
        recv_from_master(sp.moment<1>());
        recv_from_master(sp.moment<2>());
    }
}

template <class T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, int>>
auto WorkerDelegate::recv_from_master(T &&buffer) const -> T &&
{
    return std::forward<T>(buffer = comm.recv<std::decay_t<T>>(master->comm.rank));
}
template <class T, long N>
void WorkerDelegate::recv_from_master(GridArray<T, N, Pad> &buffer) const
{
    comm.recv<GridArray<T, N, Pad> const *>(master->comm.rank).unpack([&buffer](auto payload) {
        buffer = *payload;
    });
}

void WorkerDelegate::reduce_to_master(Real const &payload) const
{
    comm.send(payload, master->comm.rank).wait();
}
template <class T, long N>
void WorkerDelegate::reduce_to_master(GridArray<T, N, Pad> const &payload) const
{
    comm.send(&payload, master->comm.rank).wait(); // must wait for delivery receipt
}
HYBRID1D_END_NAMESPACE
