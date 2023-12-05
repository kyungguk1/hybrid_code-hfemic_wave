/*
 * Copyright (c) 2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "DistributedParticleDelegate.h"

#include <stdexcept>
#include <utility>

HYBRID1D_BEGIN_NAMESPACE
DistributedParticleDelegate::DistributedParticleDelegate(parallel::mpi::Comm _comm, Delegate const *subdomain_delegate)
: comm{ std::move(_comm) }, subdomain_delegate{ subdomain_delegate }
{
    if (!comm->operator bool())
        throw std::invalid_argument{ __PRETTY_FUNCTION__ };

    using parallel::mpi::ReduceOp;
    reduce_plus = {
        ReduceOp::plus<Scalar>(true),
        ReduceOp::plus<CartVector>(true),
        ReduceOp::plus<CartTensor>(true),
    };
}

void DistributedParticleDelegate::prologue(Domain const &domain, long const i) const
{
    subdomain_delegate->prologue(domain, i);
}
void DistributedParticleDelegate::epilogue(Domain const &domain, long const i) const
{
    subdomain_delegate->epilogue(domain, i);
}
void DistributedParticleDelegate::once(Domain &domain) const
{
    subdomain_delegate->once(domain);
}
void DistributedParticleDelegate::partition(PartSpecies &sp, BucketBuffer &buffer) const
{
    subdomain_delegate->partition(sp, buffer);
}
void DistributedParticleDelegate::boundary_pass(PartSpecies const &sp, BucketBuffer &buffer) const
{
    subdomain_delegate->boundary_pass(sp, buffer);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, PartSpecies &sp) const
{
    subdomain_delegate->boundary_pass(domain, sp);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, ColdSpecies &sp) const
{
    subdomain_delegate->boundary_pass(domain, sp);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, BField &bfield) const
{
    subdomain_delegate->boundary_pass(domain, bfield);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, EField &efield) const
{
    subdomain_delegate->boundary_pass(domain, efield);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, Charge &charge) const
{
    subdomain_delegate->boundary_pass(domain, charge);
}
void DistributedParticleDelegate::boundary_pass(Domain const &domain, Current &current) const
{
    subdomain_delegate->boundary_pass(domain, current);
}
void DistributedParticleDelegate::boundary_gather(Domain const &domain, Charge &charge) const
{
    subdomain_delegate->boundary_gather(domain, charge);
    accumulate_distribute<0>(charge);
}
void DistributedParticleDelegate::boundary_gather(Domain const &domain, Current &current) const
{
    subdomain_delegate->boundary_gather(domain, current);
    accumulate_distribute<1>(current);
}
void DistributedParticleDelegate::boundary_gather(Domain const &domain, Species &sp) const
{
    subdomain_delegate->boundary_gather(domain, sp);
    {
        accumulate_distribute<0>(sp.moment<0>());
        accumulate_distribute<1>(sp.moment<1>());
        accumulate_distribute<2>(sp.moment<2>());
    }
}

template <unsigned I, class T, long S>
void DistributedParticleDelegate::accumulate_distribute(GridArray<T, S, Pad> &grid) const
{
    comm.all_reduce<I>(std::get<I>(reduce_plus), grid.begin(), grid.end());
}
HYBRID1D_END_NAMESPACE
