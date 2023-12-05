/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "ParamSet.h"

#include <map>
#include <stdexcept>
#include <variant>

HYBRID1D_BEGIN_NAMESPACE
ParamSet::ParamSet(long const subdomain_rank, Options const &opts)
: geomtr{ Input::xi, Input::Dx, Input::O0 }
{
    if (subdomain_rank < 0 || subdomain_rank >= Input::number_of_subdomains)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid rank" };

    constexpr auto full_to_half_grid_shift = 0.5;
    // whole domain extent
    full_grid_whole_domain_extent = { -0.5 * Input::Nx, Input::Nx };
    half_grid_whole_domain_extent = full_grid_whole_domain_extent + full_to_half_grid_shift;

    if (!geomtr.is_valid(CurviCoord{ full_grid_whole_domain_extent.min() })
        || !geomtr.is_valid(CurviCoord{ full_grid_whole_domain_extent.max() })
        || !geomtr.is_valid(CurviCoord{ half_grid_whole_domain_extent.min() })
        || !geomtr.is_valid(CurviCoord{ half_grid_whole_domain_extent.max() }))
        throw std::invalid_argument(std::string{ __PRETTY_FUNCTION__ } + " - invalid domain extent");

    // subdomain extent
    auto const Mx     = Input::Nx / Input::number_of_subdomains;
    auto const offset = subdomain_rank * Mx;

    full_grid_subdomain_extent = { full_grid_whole_domain_extent.min() + offset, Mx };
    half_grid_subdomain_extent = full_grid_subdomain_extent + full_to_half_grid_shift;

    // optional parameters
    //
    std::map<std::string_view, std::variant<int *, long *, bool *, std::string *>> const map{
        { "wd", &working_directory },
        { "outer_Nt", &outer_Nt },
        { "save", &snapshot_save },
        { "load", &snapshot_load },
        { "record_particle_at_init", &record_particle_at_init },
        { "energy_recording_frequency", &energy_recording_frequency.first },
        { "field_recording_frequency", &field_recording_frequency.first },
        { "moment_recording_frequency", &moment_recording_frequency.first },
        { "particle_recording_frequency", &particle_recording_frequency.first },
        { "vhistogram_recording_frequency", &vhistogram_recording_frequency.first },
    };
    for (auto const &[key, val] : *opts) {
        std::visit(val, map.at(key));
    }
}

namespace {
template <class Object>
decltype(auto) write_attr(Object &obj, ParamSet const &params)
{
    using hdf5::make_type;
    using hdf5::Space;
    { // global parameters
        obj.attribute("full_grid_domain_extent", make_type(params.full_grid_whole_domain_extent.minmax()), Space::scalar())
            .write(params.full_grid_whole_domain_extent.minmax());
        obj.attribute("half_grid_domain_extent", make_type(params.half_grid_whole_domain_extent.minmax()), Space::scalar())
            .write(params.half_grid_whole_domain_extent.minmax());
        obj.attribute("number_of_worker_threads", make_type(params.number_of_worker_threads), Space::scalar())
            .write(params.number_of_worker_threads);
        obj.attribute("number_of_subdomains", make_type(params.number_of_subdomains), Space::scalar())
            .write(params.number_of_subdomains);
        obj.attribute("number_of_distributed_particle_subdomain_clones", make_type(params.number_of_distributed_particle_subdomain_clones), Space::scalar())
            .write(params.number_of_distributed_particle_subdomain_clones);
        obj.attribute("number_of_mpi_processes", make_type(params.number_of_mpi_processes), Space::scalar())
            .write(params.number_of_mpi_processes);
        obj.attribute("number_of_particle_parallelism", make_type(params.number_of_particle_parallelism), Space::scalar())
            .write(params.number_of_particle_parallelism);
        obj.attribute("algorithm", make_type<long>(params.algorithm), Space::scalar())
            .template write<long>(params.algorithm);
        obj.attribute("n_subcycles", make_type(params.n_subcycles), Space::scalar())
            .write(params.n_subcycles);
        obj.attribute("should_randomize_gyrophase_of_reflecting_particles", make_type<int>(), Space::scalar())
            .template write<int>(params.should_randomize_gyrophase_of_reflecting_particles);
        obj.attribute("particle_boundary_condition", make_type<long>(), Space::scalar())
            .write(long(params.particle_boundary_condition));
        obj.attribute("masking_inset", make_type(params.phase_retardation.masking_inset), Space::scalar())
            .write(params.phase_retardation.masking_inset);
        obj.attribute("phase_retardation", make_type(params.phase_retardation.masking_factor), Space::scalar())
            .write(params.phase_retardation.masking_factor);
        obj.attribute("amplitude_damping", make_type(params.amplitude_damping.masking_factor), Space::scalar())
            .write(params.amplitude_damping.masking_factor);
        obj.attribute("c", make_type(params.c), Space::scalar()).write(params.c);
        obj.attribute("O0", make_type(params.O0), Space::scalar()).write(params.O0);
        obj.attribute("xi", make_type(params.xi), Space::scalar()).write(params.xi);
        obj.attribute("Dx", make_type(params.Dx), Space::scalar()).write(params.Dx);
        obj.attribute("Nx", make_type(params.Nx), Space::scalar()).write(params.Nx);
        obj.attribute("dt", make_type(params.dt), Space::scalar()).write(params.dt);
        obj.attribute("innerNt", make_type(params.inner_Nt), Space::scalar())
            .write(params.inner_Nt);
        obj.attribute("partNs", make_type<long>(), Space::scalar())
            .template write<long>(std::tuple_size_v<decltype(params.part_descs)>);
        obj.attribute("coldNs", make_type<long>(), Space::scalar())
            .template write<long>(std::tuple_size_v<decltype(params.cold_descs)>);
        obj.attribute("efluid.beta", make_type(params.efluid_desc.beta), Space::scalar())
            .write(params.efluid_desc.beta);
        obj.attribute("efluid.gamma", make_type(params.efluid_desc.gamma), Space::scalar())
            .write(params.efluid_desc.gamma);
        obj.attribute("efluid.Oc", make_type(params.efluid_desc.Oc), Space::scalar())
            .write(params.efluid_desc.Oc);
        obj.attribute("efluid.op", make_type(params.efluid_desc.op), Space::scalar())
            .write(params.efluid_desc.op);
    }
    return obj;
}
} // namespace
auto operator<<(hdf5::Group &obj, ParamSet const &params) -> decltype(obj)
{
    return write_attr(obj, params);
}
auto operator<<(hdf5::Dataset &obj, ParamSet const &params) -> decltype(obj)
{
    return write_attr(obj, params);
}
HYBRID1D_END_NAMESPACE
