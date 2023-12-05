/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Delegate.h"

#include <algorithm>
#include <cmath>
#include <random>

HYBRID1D_BEGIN_NAMESPACE
auto Delegate::BucketBuffer::cleared() -> decltype(auto)
{
    L.clear();
    R.clear();
    return (*this);
}
auto Delegate::bucket_buffer() const -> BucketBuffer &
{
    return m_buckets->cleared();
}

Delegate::~Delegate()
{
}
Delegate::Delegate() noexcept
: m_buckets{ std::make_unique<BucketBuffer>() }
{
}

void Delegate::partition(PartSpecies &sp, BucketBuffer &buffer) const
{
    auto const &subdomain_extent = sp.particle_subdomain_extent();

    // group particles that have crossed left boundary
    auto L_it = std::partition(sp.bucket.begin(), sp.bucket.end(), [LB = subdomain_extent.min()](Particle const &ptl) noexcept -> bool {
        return ptl.pos.q1 >= LB;
    });
    buffer.L.insert(buffer.L.cend(), L_it, sp.bucket.end());
    sp.bucket.erase(L_it, sp.bucket.end());

    // group particles that have crossed right boundary
    auto R_it = std::partition(sp.bucket.begin(), sp.bucket.end(), [RB = subdomain_extent.max()](Particle const &ptl) noexcept -> bool {
        return ptl.pos.q1 < RB;
    });
    buffer.R.insert(buffer.R.cend(), R_it, sp.bucket.end());
    sp.bucket.erase(R_it, sp.bucket.end());
}
void Delegate::boundary_pass(PartSpecies const &sp, BucketBuffer &buffer) const
{
    switch (sp.params.particle_boundary_condition) {
        case BC::periodic:
            periodic_particle_pass(sp, buffer);
            break;
        case BC::reflecting:
            reflecting_particle_pass(sp, buffer);
            break;
    }
    buffer.swap_bucket();
}
void Delegate::boundary_pass(Domain const &, PartSpecies &sp) const
{
    // be careful not to access it from multiple threads
    // note that the content is cleared after this call
    auto &buffer = bucket_buffer();
    //
    partition(sp, buffer);
    boundary_pass(sp, buffer);
    //
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.L), cend(buffer.L));
    sp.bucket.insert(sp.bucket.cend(), cbegin(buffer.R), cend(buffer.R));
}

template <class T, long N>
void Delegate::pass(GridArray<T, N, Pad> &A)
{
    // fill ghost cells
    //
    for (long p = 0, m = -1; p < Pad; ++p, --m) {
        // put left boundary value to right ghost cell
        A.end()[p] = A[p];
        // put right boundary value to left ghost cell
        A[m] = A.end()[m];
    }
}
template <class T, long N>
void Delegate::gather(GridArray<T, N, Pad> &A)
{
    // gather moments at ghost cells
    //
    for (long p = Pad - 1, m = -Pad; m < 0; --p, ++m) {
        // add right ghost cell value to left boundary
        A[p] += A.end()[p];
        // add left ghost cell value to right boundary
        A.end()[m] += A[m];
    }
}

void Delegate::periodic_particle_pass(PartSpecies const &sp, BucketBuffer &buffer)
{
    // adjust coordinates
    auto const &whole_domain_extent = sp.particle_whole_domain_extent();
    for (Particle &ptl : buffer.L) {
        // crossed left boundary; wrap around to the rightmost cell
        if (!whole_domain_extent.is_member(ptl.pos.q1))
            ptl.pos.q1 += whole_domain_extent.len;
    }
    for (Particle &ptl : buffer.R) {
        // crossed right boundary; wrap around to the leftmost cell
        if (!whole_domain_extent.is_member(ptl.pos.q1))
            ptl.pos.q1 -= whole_domain_extent.len;
    }
}
void Delegate::reflecting_particle_pass(PartSpecies const &sp, BucketBuffer &buffer)
{
    // adjust coordinates and flip velocity vector
    auto const &whole_domain_extent = sp.particle_whole_domain_extent();
    auto const &geomtr              = sp.geomtr;
    for (Particle &ptl : buffer.L) {
        if (!whole_domain_extent.is_member(ptl.pos.q1)) {
            // crossed left boundary; put back into the leftmost cell
            ptl.pos.q1 += 2 * (whole_domain_extent.min() - ptl.pos.q1);
            // flip the velocity component parallel to B0
            CartVector const e1 = geomtr.e1(ptl.pos);
            ptl.vel -= 2 * dot(ptl.vel, e1) * e1;
            // gyro-phase randomization
            if constexpr (ParamSet::should_randomize_gyrophase_of_reflecting_particles)
                ptl.vel = geomtr.mfa_to_cart(randomize_gyrophase(geomtr.cart_to_mfa(ptl.vel, ptl.pos)), ptl.pos);
        }
    }
    for (Particle &ptl : buffer.R) {
        if (!whole_domain_extent.is_member(ptl.pos.q1)) {
            // crossed right boundary; put back into the rightmost cell
            ptl.pos.q1 += 2 * (whole_domain_extent.max() - ptl.pos.q1);
            // flip the velocity component parallel to B0
            CartVector const e1 = geomtr.e1(ptl.pos);
            ptl.vel -= 2 * dot(ptl.vel, e1) * e1;
            // gyro-phase randomization
            if constexpr (ParamSet::should_randomize_gyrophase_of_reflecting_particles)
                ptl.vel = geomtr.mfa_to_cart(randomize_gyrophase(geomtr.cart_to_mfa(ptl.vel, ptl.pos)), ptl.pos);
        }
    }
}
auto Delegate::randomize_gyrophase(MFAVector const &v) -> MFAVector
{
    thread_local static std::mt19937                     rng{ std::random_device{}() };
    thread_local static std::uniform_real_distribution<> dist{ -M_PI, M_PI };

    auto const phase = dist(rng);
    auto const cos   = std::cos(phase);
    auto const sin   = std::sin(phase);

    return { v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos };
}
HYBRID1D_END_NAMESPACE
