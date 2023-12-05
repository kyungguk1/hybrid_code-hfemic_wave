/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Boundary/Delegate.h"
#include "Boundary/DistributedParticleDelegate.h"
#include "Boundary/MasterDelegate.h"
#include "Boundary/SubdomainDelegate.h"
#include "Core/Domain.h"
#include "ParamSet.h"
#include "Recorder/Recorder.h"

#include <ParallelKit/ParallelKit.h>
#include <array>
#include <future>
#include <map>
#include <memory>
#include <string>

HYBRID1D_BEGIN_NAMESPACE
class [[nodiscard]] Driver {
    long                                             iteration_count{};
    parallel::mpi::Comm const                        world;
    ParamSet                                         params;
    parallel::mpi::Comm                              subdomain_comm;
    parallel::mpi::Comm                              distributed_particle_comm;
    std::unique_ptr<Domain>                          domain;
    std::unique_ptr<MasterDelegate>                  master;
    std::unique_ptr<SubdomainDelegate>               subdomain_delegate;
    std::unique_ptr<DistributedParticleDelegate>     distributed_particle_delegate;
    std::map<std::string, std::unique_ptr<Recorder>> recorders;

    struct [[nodiscard]] Worker {
        long                    iteration_count;
        Driver const           *driver;
        WorkerDelegate         *delegate;
        std::future<void>       handle;
        std::unique_ptr<Domain> domain;
        //
        void operator()();
        Worker() noexcept          = default;
        Worker(Worker &&) noexcept = default;
    };
    std::array<Worker, ParamSet::number_of_particle_parallelism - 1> workers;

public:
    ~Driver();
    Driver(parallel::mpi::Comm comm, Options const &opts);
    Driver(Driver const &) = delete;

    void operator()();

private:
    void master_loop();

    [[nodiscard]] static std::unique_ptr<Domain> make_domain(ParamSet const &params, Delegate *delegate);
};
HYBRID1D_END_NAMESPACE
