/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "SubdomainDelegate.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <stdexcept>
#include <utility>

HYBRID1D_BEGIN_NAMESPACE
SubdomainDelegate::SubdomainDelegate(parallel::mpi::Comm _comm)
: comm{ std::move(_comm) }
{
    if (!comm->operator bool())
        throw std::invalid_argument{ __PRETTY_FUNCTION__ };

    int const size = comm.size();
    int const rank = comm->rank();
    left_          = rank_t{ (size + rank - 1) % size };
    right          = rank_t{ (size + rank + 1) % size };
}

void SubdomainDelegate::once(Domain &domain) const
{
    std::mt19937                     g{ 494983U + static_cast<unsigned>(comm->rank()) };
    std::uniform_real_distribution<> d{ -1, 1 };
    for (auto &v : domain.bfield) {
        v.x += d(g) * Debug::initial_bfield_noise_amplitude;
        v.y += d(g) * Debug::initial_bfield_noise_amplitude;
        v.z += d(g) * Debug::initial_bfield_noise_amplitude;
    }
}

void SubdomainDelegate::boundary_pass(Domain const &, ColdSpecies &sp) const
{
    mpi_pass(sp.mom0_full);
    mpi_pass(sp.mom1_full);
}
void SubdomainDelegate::boundary_pass(Domain const &, BField &bfield) const
{
    if constexpr (Debug::zero_out_electromagnetic_field) {
        bfield.fill_all(CartVector{});
    }
    mpi_pass(bfield);
}
void SubdomainDelegate::boundary_pass(Domain const &, EField &efield) const
{
    if constexpr (Debug::zero_out_electromagnetic_field) {
        efield.fill_all(CartVector{});
    }
    mpi_pass(efield);
}
template <class T, long Mx>
void SubdomainDelegate::mpi_pass(GridArray<T, Mx, Pad> &grid) const
{
    // pass across boundaries
    // send-recv pair order is important
    // e.g., if send-left is first, recv-right should appear first.

    constexpr parallel::mpi::Tag tag1{ 1 };
    constexpr parallel::mpi::Tag tag2{ 2 };
    if constexpr (Mx >= Pad) {
        auto tk_left_ = comm.issend<T>(grid.begin(), std::next(grid.begin(), Pad), { left_, tag1 });
        auto tk_right = comm.issend<T>(std::prev(grid.end(), Pad), grid.end(), { right, tag2 });
        {
            comm.recv<T>(grid.end(), std::next(grid.end(), Pad), { right, tag1 });
            comm.recv<T>(std::prev(grid.begin(), Pad), grid.begin(), { left_, tag2 });
        }
        std::move(tk_left_).wait();
        std::move(tk_right).wait();
    } else {
        // from inside out
        //
        for (long b = 0, e = -1; b < Pad; ++b, --e) {
            auto tk_left_ = comm.issend<T>(grid.begin()[b], { left_, tag1 });
            auto tk_right = comm.issend<T>(grid.end()[e], { right, tag2 });
            {
                grid.end()[b]   = comm.recv<T>({ right, tag1 });
                grid.begin()[e] = comm.recv<T>({ left_, tag2 });
            }
            std::move(tk_left_).wait();
            std::move(tk_right).wait();
        }
    }
}

void SubdomainDelegate::boundary_pass(Domain const &, Charge &charge) const
{
    mpi_pass(charge);
}
void SubdomainDelegate::boundary_pass(Domain const &, Current &current) const
{
    mpi_pass(current);
}
void SubdomainDelegate::boundary_gather(Domain const &, Charge &charge) const
{
    moment_gather(charge.params, charge);

    mpi_gather(charge);
}
void SubdomainDelegate::boundary_gather(Domain const &, Current &current) const
{
    moment_gather(current.params, current);

    mpi_gather(current);
}
void SubdomainDelegate::boundary_gather(Domain const &, Species &sp) const
{
    moment_gather(sp.params, sp.moment<0>());
    moment_gather(sp.params, sp.moment<1>());
    moment_gather(sp.params, sp.moment<2>());

    mpi_gather(sp.moment<0>());
    mpi_gather(sp.moment<1>());
    mpi_gather(sp.moment<2>());
}
template <class T, long Mx>
void SubdomainDelegate::moment_gather(ParamSet const &params, GridArray<T, Mx, Pad> &grid) const
{
    switch (params.particle_boundary_condition) {
        case BC::periodic:
            // do nothing
            break;
        case BC::reflecting: {
            if (is_leftmost_subdomain()) {
                // at the leftmost subdomain, moments are accumulated at the first grid point
                for (long i = -Pad; i < 0; ++i) {
                    grid[i + 1] += std::exchange(grid[i], T{});
                }
            }
            if (is_rightmost_subdomain()) {
                // at the rightmost subdomain, moments are accumulated at the last grid point
                for (long i = Mx + Pad - 1; i >= Mx; --i) {
                    grid[i - 1] += std::exchange(grid[i], T{});
                }
            }
            break;
        }
    }
}
template <class T, long Mx>
void SubdomainDelegate::mpi_gather(GridArray<T, Mx, Pad> &grid) const
{
    // pass across boundaries
    // send-recv pair order is important
    // e.g., if send-left is first, recv-right should appear first.

    constexpr parallel::mpi::Tag tag1{ 1 };
    constexpr parallel::mpi::Tag tag2{ 2 };
    if constexpr (Mx >= Pad) {
        auto tk_left_ = comm.issend<T>(std::prev(grid.begin(), Pad), grid.begin(), { left_, tag1 });
        auto tk_right = comm.issend<T>(grid.end(), std::next(grid.end(), Pad), { right, tag2 });
        {
            auto const accum = [](auto payload, auto *first, auto *last) {
                std::transform(first, last, begin(payload), first, std::plus{});
            };
            comm.recv<T>({}, { right, tag1 }).unpack(accum, std::prev(grid.end(), Pad), grid.end());
            comm.recv<T>({}, { left_, tag2 }).unpack(accum, grid.begin(), std::next(grid.begin(), Pad));
        }
        std::move(tk_left_).wait();
        std::move(tk_right).wait();
    } else {
        // from outside in
        //
        for (long b = -Pad, e = Pad - 1; b < 0; ++b, --e) {
            auto tk_left_ = comm.issend<T>(grid.begin()[b], { left_, tag1 });
            auto tk_right = comm.issend<T>(grid.end()[e], { right, tag2 });
            {
                grid.end()[b] += comm.recv<T>({ right, tag1 });
                grid.begin()[e] += comm.recv<T>({ left_, tag2 });
            }
            std::move(tk_left_).wait();
            std::move(tk_right).wait();
        }
    }
}

void SubdomainDelegate::boundary_pass(PartSpecies const &sp, BucketBuffer &buffer) const
{
    // pass particles across subdomain boundaries
    //
    switch (sp.params.particle_boundary_condition) {
        case BC::periodic:
            periodic_particle_pass(sp, buffer);
            break;
        case BC::reflecting:
            reflecting_particle_pass(sp, buffer);
            break;
    }

    // adjust coordinates and (if necessary) flip velocity vector
    //
    Delegate::boundary_pass(sp, buffer);
}
void SubdomainDelegate::periodic_particle_pass(PartSpecies const &, BucketBuffer &buffer) const
{
    // pass particles across boundaries
    //
    mpi_pass(buffer);
}
void SubdomainDelegate::reflecting_particle_pass(PartSpecies const &, BucketBuffer &buffer) const
{
    // hijack boundary-crossing particles
    //
    BucketBuffer hijacked;
    if (is_leftmost_subdomain()) {
        hijacked.L = std::move(buffer.L);
        buffer.L.clear();
    } else {
        hijacked.L.clear();
    }
    if (is_rightmost_subdomain()) {
        hijacked.R = std::move(buffer.R);
        buffer.R.clear();
    } else {
        hijacked.R.clear();
    }

    // pass remaining particles across subdomain boundaries
    //
    mpi_pass(buffer);

    // put back the hijacked particles
    //
    buffer.L.insert(end(buffer.L), std::make_move_iterator(begin(hijacked.L)), std::make_move_iterator(end(hijacked.L)));
    buffer.R.insert(end(buffer.R), std::make_move_iterator(begin(hijacked.R)), std::make_move_iterator(end(hijacked.R)));
}
void SubdomainDelegate::mpi_pass(BucketBuffer &buffer) const
{
    // pass particles across boundaries
    // send-recv pair order is important
    // e.g., if send-left is first, recv-right should appear first.
    //
    constexpr parallel::mpi::Tag tag1{ 1 };
    constexpr parallel::mpi::Tag tag2{ 2 };
    {
        auto tk1 = comm.ibsend(std::move(buffer.L), { left_, tag1 });
        auto tk2 = comm.ibsend(std::move(buffer.R), { right, tag2 });
        {
            buffer.L = comm.recv<3>({}, { right, tag1 });
            buffer.R = comm.recv<3>({}, { left_, tag2 });
        }
        std::move(tk1).wait();
        std::move(tk2).wait();
    }
}
HYBRID1D_END_NAMESPACE
