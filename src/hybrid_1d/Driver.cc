/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Driver.h"
#include "Core/Domain_CAMCL.h"
#include "Core/Domain_PC.h"
#include "Recorder/EnergyRecorder.h"
#include "Recorder/FieldRecorder.h"
#include "Recorder/MomentRecorder.h"
#include "Recorder/ParticleRecorder.h"
#include "Recorder/Snapshot.h"
#include "Recorder/SnapshotGrid.h"
#include "Recorder/SnapshotParticle.h"
#include "Recorder/VHistogramRecorder.h"
#include <PIC/UTL/lippincott.h>
#include <PIC/UTL/println.h>

#include <chrono>
#include <exception>
#include <iostream>

HYBRID1D_BEGIN_NAMESPACE
namespace {
template <class F, class... Args>
[[nodiscard]] auto measure(F &&f, Args &&...args) -> std::chrono::duration<double>
{
    static_assert(std::is_invocable_v<F &&, Args &&...>);
    auto const start = std::chrono::steady_clock::now();
    {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
    auto const end = std::chrono::steady_clock::now();

    return end - start;
}
} // namespace

Driver::~Driver()
{
}
Driver::Driver(parallel::mpi::Comm _comm, Options const &opts)
: world{ std::move(_comm) }
{
    try {
        if (!world)
            fatal_error(__PRETTY_FUNCTION__, " - invalid mpi::Comm object");

        if (auto const size = world.size(); size != ParamSet::number_of_mpi_processes)
            fatal_error(__PRETTY_FUNCTION__, " - the mpi world size (= ", std::to_string(size),
                        ") is not the same as ParamSet::number_of_mpi_processes (= ", std::to_string(ParamSet::number_of_mpi_processes), ')');

        auto const world_rank = world.rank();

        // group comm's
        // say, there are 3 subdomains
        // then the grouping of subdomain_comm's are {0, 1, 2}, {3, 4, 5}, ...
        // and the grouping of distributed_particle_comm's are {0, 3, 6, ...}, {1, 4, 7, ...}, and {2, 5, 8, ...}
        //
        subdomain_comm            = world.split(world_rank / long{ ParamSet::number_of_subdomains });
        distributed_particle_comm = world.split(world_rank % long{ ParamSet::number_of_subdomains });

        // init ParamSet
        //
        params = { subdomain_comm.rank(), opts };

        // init recorders
        //
        recorders["energy"]    = std::make_unique<EnergyRecorder>(params, subdomain_comm.duplicated(), world);
        recorders["fields"]    = std::make_unique<FieldRecorder>(params, subdomain_comm.duplicated(), world);
        recorders["moment"]    = std::make_unique<MomentRecorder>(params, subdomain_comm.duplicated(), world);
        recorders["vhists"]    = std::make_unique<VHistogramRecorder>(params, subdomain_comm.duplicated(), world);
        recorders["particles"] = std::make_unique<ParticleRecorder>(params, subdomain_comm.duplicated(), world);

        // init delegates
        //
        subdomain_delegate            = std::make_unique<SubdomainDelegate>(subdomain_comm.duplicated());
        distributed_particle_delegate = std::make_unique<DistributedParticleDelegate>(distributed_particle_comm.duplicated(), subdomain_delegate.get());
        master                        = std::make_unique<MasterDelegate>(distributed_particle_delegate.get());

        // init domain
        //
        if (0 == world_rank)
            println(std::cout, __FUNCTION__, "> initializing domain(s)");
        domain = make_domain(params, master.get());

        // init particles or load snapshot
        //
        if (params.snapshot_load) {
            if (0 == world_rank)
                print(std::cout, "\tloading snapshots") << std::endl;
            if constexpr (Debug::should_use_unified_snapshot) {
                iteration_count = load(Snapshot{ subdomain_comm.duplicated(), params, distributed_particle_comm.rank() }, *domain);
            } else {
                iteration_count = load(SnapshotGrid{ subdomain_comm.duplicated(), params }, distributed_particle_comm, *domain);
                iteration_count = load(SnapshotParticle{ world.duplicated(), params }, distributed_particle_comm, *domain);
                world.barrier();
            }
        } else {
            long species_id = 0;
            for (PartSpecies &sp : domain->part_species) {
                if (0 == world_rank)
                    print(std::cout, "\tinitializing ", species_id++ + 1, "th particles") << std::endl;
                sp.populate(distributed_particle_comm.rank(), distributed_particle_comm.size());
            }
            for (ColdSpecies &sp : domain->cold_species) {
                if (0 == world_rank)
                    print(std::cout, "\tinitializing ", species_id++ + 1, "th particles") << std::endl;
                sp.populate(distributed_particle_comm.rank(), distributed_particle_comm.size());
            }

            if (params.record_particle_at_init) {
                // first, collect particle moments
                for (PartSpecies &sp : domain->part_species) {
                    sp.collect_all();
                    master->delegate->boundary_gather(*domain, sp);
                }

                // then, dump
                if (auto const &recorder = recorders.at("particles"))
                    recorder->record(*domain, iteration_count);
                if (auto const &recorder = recorders.at("vhists"))
                    recorder->record(*domain, iteration_count);
            }
        }

        // set step count of the external sources
        for (ExternalSource &sp : domain->external_sources) {
            sp.set_cur_step(iteration_count);
        }
    } catch (std::exception const &e) {
        fatal_error(__PRETTY_FUNCTION__, " :: ", e.what());
    }
}
auto Driver::make_domain(ParamSet const &params, Delegate *delegate) -> std::unique_ptr<Domain>
{
    switch (Input::algorithm) {
        case PC:
            return std::make_unique<Domain_PC>(params, delegate);
            break;
        case CAMCL:
            return std::make_unique<Domain_CAMCL>(params, delegate);
            break;
    }
}

void Driver::operator()()
try {
    // worker setup
    //
    for (unsigned i = 0; i < workers.size(); ++i) {
        Worker &worker         = workers[i];
        worker.driver          = this;
        worker.iteration_count = iteration_count;
        worker.delegate        = &master->workers.at(i);
        worker.domain          = make_domain(params, worker.delegate);
        worker.handle          = std::async(std::launch::async, worker.delegate->wrap_loop(std::ref(worker)), worker.domain.get());
    }

    // master loop
    //
    auto const elapsed = measure(master->wrap_loop(&Driver::master_loop, this), this->domain.get());
    if (0 == world.rank())
        println(std::cout, "%% time elapsed: ", elapsed.count(), 's');

    // worker teardown
    //
    for (Worker &worker : workers) {
        worker.handle.get();
        worker.domain.reset();
    }

    // take snapshot
    //
    if (params.snapshot_save) {
        if (0 == world.rank())
            print(std::cout, "\tsaving snapshots") << std::endl;
        if constexpr (Debug::should_use_unified_snapshot) {
            save(Snapshot{ subdomain_comm.duplicated(), params, distributed_particle_comm.rank() }, *domain, iteration_count);
        } else {
            save(SnapshotGrid{ subdomain_comm.duplicated(), params }, distributed_particle_comm, *domain, iteration_count);
            save(SnapshotParticle{ world.duplicated(), params }, distributed_particle_comm, *domain, iteration_count);
            world.barrier();
        }
    }
} catch (std::exception const &e) {
    fatal_error(__PRETTY_FUNCTION__, " :: ", e.what());
}
void Driver::master_loop()
try {
    for (long outer_step = 1; outer_step <= params.outer_Nt; ++outer_step) {
        if (0 == world.rank()) {
            print(std::cout, __FUNCTION__, "> ",
                  "steps(x", params.inner_Nt, ") = ", outer_step, "/", params.outer_Nt,
                  "; time = ", iteration_count * params.dt);
            std::cout << std::endl;
        }

        // inner loop
        //
        domain->advance_by(params.inner_Nt);

        // increment step count
        //
        iteration_count += params.inner_Nt;

        // record data
        //
        if (auto const &vhists = this->recorders.at("vhists"), &particles = this->recorders.at("particles");
            (vhists && vhists->should_record_at(iteration_count)) || (particles && particles->should_record_at(iteration_count))) {
            // particle collection needed
            //
            master.get()->guarded_record([this] {
                for (auto &pair : recorders) {
                    if (pair.second)
                        pair.second->record(*domain, iteration_count);
                }
            })(*domain);
        } else {
            // no particle collection needed
            //
            for (auto &pair : recorders) {
                if (pair.second)
                    pair.second->record(*domain, iteration_count);
            }
        }
    }
} catch (std::exception const &e) {
    fatal_error(__PRETTY_FUNCTION__, " :: ", e.what());
}
void Driver::Worker::operator()()
try {
    for (long outer_step = 1; outer_step <= domain->params.outer_Nt; ++outer_step) {
        // inner loop
        //
        domain->advance_by(domain->params.inner_Nt);

        // increment step count
        //
        iteration_count += domain->params.inner_Nt;

        // record data
        //
        if (auto const &vhists = driver->recorders.at("vhists"), &particles = driver->recorders.at("particles");
            (vhists && vhists->should_record_at(iteration_count)) || (particles && particles->should_record_at(iteration_count))) {
            // particle collection needed
            //
            delegate->guarded_record(*domain);
        } else {
            // no particle collection needed
            //
        }
    }
} catch (std::exception const &e) {
    fatal_error(__PRETTY_FUNCTION__, " :: ", e.what());
}
HYBRID1D_END_NAMESPACE
