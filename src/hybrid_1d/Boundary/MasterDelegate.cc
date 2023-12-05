/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "MasterDelegate.h"

#include <algorithm>
#include <iterator>

HYBRID1D_BEGIN_NAMESPACE
MasterDelegate::~MasterDelegate()
{
}
MasterDelegate::MasterDelegate(Delegate *const delegate)
: delegate{ delegate }, all_but_master{}
{
    comm = dispatch.comm(static_cast<unsigned>(workers.size()));
    for (unsigned i = 0; i < workers.size(); ++i) {
        workers[i].master = this;
        workers[i].comm   = dispatch.comm(i);
        all_but_master.emplace_back(i);
    }
}

void MasterDelegate::setup(Domain &domain) const
{
    if (auto const divisor = workers.size() + 1; !workers.empty()) {
        // distribute particles to workers
        //
        for (PartSpecies &sp : domain.part_species) {
            // distribute moment weighting factor to workers
            broadcast_to_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}) /= divisor);

            // distribute particles to workers
            distribute_to_workers(sp, sp.bucket);
        }

        // distribute cold species to workers
        //
        for (ColdSpecies &sp : domain.cold_species) {
            // distribute moment weighting factor to workers
            broadcast_to_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}) /= divisor);
        }

        // distribute external sources to workers
        //
        for (ExternalSource &sp : domain.external_sources) {
            // distribute moment weighting factor to workers
            broadcast_to_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}) /= divisor);
        }
    }
}
template <class Container>
void MasterDelegate::distribute_to_workers(PartSpecies const &, Container &bucket) const
{
    std::vector<Container> payloads;
    payloads.reserve(all_but_master.size());
    auto const chunk = static_cast<long>(bucket.size() / (workers.size() + 1));
    for ([[maybe_unused]] rank_t const &rank : all_but_master) { // master excluded
        auto const last  = end(bucket);
        auto const first = std::prev(last, chunk);
        payloads.emplace_back(std::make_move_iterator(first), std::make_move_iterator(last));
        bucket.erase(first, last);
    }
    auto tks = comm.scatter(std::move(payloads), all_but_master);
    std::for_each(std::make_move_iterator(begin(tks)), std::make_move_iterator(end(tks)),
                  std::mem_fn(&ticket_t::wait));
}

void MasterDelegate::teardown(Domain &domain) const
{
    if (!workers.empty()) {
        // collect particles from workers
        //
        for (PartSpecies &sp : domain.part_species) {
            // collect moment weighting factor from workers
            collect_from_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}));

            // collect particles from workers
            collect_from_workers(sp, sp.bucket);
        }

        // collect cold species from workers
        //
        for (ColdSpecies &sp : domain.cold_species) {
            // collect moment weighting factor from workers
            collect_from_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}));
        }

        // collect external sources to master
        //
        for (ExternalSource &sp : domain.external_sources) {
            // collect moment weighting factor from workers
            collect_from_workers(sp.moment_weighting_factor(Badge<MasterDelegate>{}));
        }
    }
}
template <class Container>
void MasterDelegate::collect_from_workers(PartSpecies const &, Container &bucket) const
{
    // gather particles from workers
    //
    comm.for_each<Container>(
        all_but_master,
        [](auto payload, Container &bucket) {
            std::move(begin(payload), end(payload), std::back_inserter(bucket));
        },
        bucket);
}

void MasterDelegate::prologue(Domain const &domain, long const i) const
{
    delegate->prologue(domain, i);
}
void MasterDelegate::epilogue(Domain const &domain, long const i) const
{
    delegate->epilogue(domain, i);
}
void MasterDelegate::once(Domain &domain) const
{
    if (!workers.empty()) {
        // distribute particles' equilibrium moments to workers
        //
        for (PartSpecies &sp : domain.part_species) {
            broadcast_to_workers(sp.equilibrium_mom0);
            broadcast_to_workers(sp.equilibrium_mom1);
            broadcast_to_workers(sp.equilibrium_mom2);
        }

        // distribute species' cold moments to workers
        //
        for (ColdSpecies &sp : domain.cold_species) {
            broadcast_to_workers(sp.mom0_full);
            broadcast_to_workers(sp.mom1_full);
        }

        // distribute external sources' current time step to workers
        //
        for (ExternalSource &sp : domain.external_sources) {
            // distribute current time step to worker
            broadcast_to_workers(sp.cur_step());
        }
    }

    delegate->once(domain);
}
void MasterDelegate::boundary_pass(Domain const &, PartSpecies &sp) const
{
    // be careful not to access it from multiple threads
    // note that the content is cleared after this call
    auto &buffer = bucket_buffer();
    delegate->partition(sp, buffer);
    //
    if (!workers.empty()) {
        collect_from_workers(sp, buffer.L);
        collect_from_workers(sp, buffer.R);
    }
    delegate->boundary_pass(sp, buffer);
    if (!workers.empty()) {
        distribute_to_workers(sp, buffer.L);
        distribute_to_workers(sp, buffer.R);
    }
    //
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.L), cend(buffer.L));
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.R), cend(buffer.R));
}
void MasterDelegate::boundary_pass(Domain const &domain, ColdSpecies &sp) const
{
    delegate->boundary_pass(domain, sp);
    broadcast_to_workers(sp.mom0_full);
    broadcast_to_workers(sp.mom1_full);
}
void MasterDelegate::boundary_pass(Domain const &domain, BField &bfield) const
{
    delegate->boundary_pass(domain, bfield);
    broadcast_to_workers(bfield);
}
void MasterDelegate::boundary_pass(Domain const &domain, EField &efield) const
{
    delegate->boundary_pass(domain, efield);
    broadcast_to_workers(efield);
}
void MasterDelegate::boundary_pass(Domain const &domain, Charge &charge) const
{
    delegate->boundary_pass(domain, charge);
    broadcast_to_workers(charge);
}
void MasterDelegate::boundary_pass(Domain const &domain, Current &current) const
{
    delegate->boundary_pass(domain, current);
    broadcast_to_workers(current);
}
void MasterDelegate::boundary_gather(Domain const &domain, Charge &charge) const
{
    collect_from_workers(charge);
    delegate->boundary_gather(domain, charge);
    broadcast_to_workers(charge);
}
void MasterDelegate::boundary_gather(Domain const &domain, Current &current) const
{
    collect_from_workers(current);
    delegate->boundary_gather(domain, current);
    broadcast_to_workers(current);
}
void MasterDelegate::boundary_gather(Domain const &domain, Species &sp) const
{
    {
        collect_from_workers(sp.moment<0>());
        collect_from_workers(sp.moment<1>());
        collect_from_workers(sp.moment<2>());
    }
    delegate->boundary_gather(domain, sp);
    {
        broadcast_to_workers(sp.moment<0>());
        broadcast_to_workers(sp.moment<1>());
        broadcast_to_workers(sp.moment<2>());
    }
}

namespace {
template <class T, long N>
decltype(auto) operator+=(GridArray<T, N, Pad> &lhs, GridArray<T, N, Pad> const &rhs) noexcept
{
    auto lhs_first = lhs.dead_begin(), lhs_last = lhs.dead_end();
    auto rhs_first = rhs.dead_begin();
    while (lhs_first != lhs_last) {
        *lhs_first++ += *rhs_first++;
    }
    return lhs;
}
} // namespace
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int>>
void MasterDelegate::broadcast_to_workers(T const &payload) const
{
    auto tks = comm.bcast<T>(payload, all_but_master);
    std::for_each(std::make_move_iterator(begin(tks)), std::make_move_iterator(end(tks)),
                  std::mem_fn(&ticket_t::wait));
}
template <class T, long N>
void MasterDelegate::broadcast_to_workers(GridArray<T, N, Pad> const &payload) const
{
    auto tks = comm.bcast(&payload, all_but_master);
    std::for_each(std::make_move_iterator(begin(tks)), std::make_move_iterator(end(tks)),
                  std::mem_fn(&ticket_t::wait));
}

void MasterDelegate::collect_from_workers(Real &buffer) const
{
    // the first worker will collect all workers'
    //
    comm.for_each<Real>(all_but_master, [&buffer](auto payload) {
        buffer += payload;
    });
}
template <class T, long N>
void MasterDelegate::collect_from_workers(GridArray<T, N, Pad> &buffer) const
{
    // the first worker will collect all workers'
    //
    comm.for_each<GridArray<T, N, Pad> const *>(
        all_but_master,
        [](auto payload, GridArray<T, N, Pad> &buffer) {
            buffer += *payload;
        },
        buffer);
}
HYBRID1D_END_NAMESPACE
