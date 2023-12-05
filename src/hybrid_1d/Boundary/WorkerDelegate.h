/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Delegate.h"

#include <ParallelKit/ParallelKit.h>
#include <functional>
#include <type_traits>

HYBRID1D_BEGIN_NAMESPACE
class MasterDelegate;

class WorkerDelegate final : public Delegate {
public:
    using message_dispatch_t
        = parallel::MessageDispatch<std::vector<Particle>, std::deque<Particle>,
                                    Grid<Scalar> const *, Grid<CartVector> const *, Grid<CartTensor> const *,
                                    long, Real>;
    using interthread_comm_t = message_dispatch_t::Communicator;
    //
    MasterDelegate    *master{};
    interthread_comm_t comm{};

private:
    void once(Domain &) const override;
    void prologue(Domain const &, long) const override;
    void epilogue(Domain const &, long) const override;
    void boundary_pass(Domain const &, PartSpecies &) const override;
    void boundary_pass(Domain const &, ColdSpecies &) const override;
    void boundary_pass(Domain const &, BField &) const override;
    void boundary_pass(Domain const &, EField &) const override;
    void boundary_pass(Domain const &, Charge &) const override;
    void boundary_pass(Domain const &, Current &) const override;
    void boundary_gather(Domain const &, Charge &) const override;
    void boundary_gather(Domain const &, Current &) const override;
    void boundary_gather(Domain const &, Species &) const override;

    // helpers
    template <class T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, int> = 0>
    auto recv_from_master(T &&) const -> T &&;
    template <class T, long N>
    void recv_from_master(GridArray<T, N, Pad> &buffer) const;

    void reduce_to_master(Real const &payload) const;
    template <class T, long N>
    void reduce_to_master(GridArray<T, N, Pad> const &payload) const;

    template <class Container>
    void collect_to_master(PartSpecies const &, Container &bucket) const;
    template <class Container>
    void recv_from_master(PartSpecies const &, Container &bucket) const;

public: // wrap the loop with setup/teardown logic included
    template <class F, class... Args>
    [[nodiscard]] auto wrap_loop(F &&f, Args &&...args)
    {
        return [this, f, args...](Domain *domain) mutable { // intentional capture by copy
            setup(*domain);
            std::invoke(std::forward<F>(f), std::move(args)...); // hence, move is used
            teardown(*domain);
        };
    }

    void guarded_record(Domain &domain)
    {
        teardown(domain);
        setup(domain);
    }

private:
    void setup(Domain &) const;
    void teardown(Domain &) const;
};
HYBRID1D_END_NAMESPACE
