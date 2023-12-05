/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "InputWrapper.h"
#include <PIC/Geometry.h>
#include <PIC/UTL/Options.h>

#include <HDF5Kit/HDF5Kit.h>
#include <string>

HYBRID1D_BEGIN_NAMESPACE
struct [[nodiscard]] ParamSet : public Input {

    /// number of mpi processes needed
    ///
    static constexpr unsigned number_of_mpi_processes = number_of_subdomains * number_of_distributed_particle_subdomain_clones;

    /// number of threads for particle async update
    ///
    static constexpr unsigned number_of_particle_parallelism = (number_of_worker_threads + 1) / number_of_mpi_processes;

    /// index sequence of kinetic plasma descriptors
    ///
    using part_indices = std::make_index_sequence<std::tuple_size_v<decltype(part_descs)>>;

    /// index sequence of cold plasma descriptors
    ///
    using cold_indices = std::make_index_sequence<std::tuple_size_v<decltype(cold_descs)>>;

    /// index sequence of external source descriptors
    ///
    using source_indices = std::make_index_sequence<std::tuple_size_v<decltype(source_descs)>>;

public:
    Geometry    geomtr;
    Range       full_grid_whole_domain_extent{ -1, 0 };
    Range       half_grid_whole_domain_extent{ -1, 0 };
    Range       full_grid_subdomain_extent{ -1, 0 };
    Range       half_grid_subdomain_extent{ -1, 0 };
    long        outer_Nt{ Input::outer_Nt };
    std::string working_directory{ Input::working_directory };
    bool        snapshot_save{ false };
    bool        snapshot_load{ false };
    bool        record_particle_at_init{ false };
    //
    std::pair<int, Range> energy_recording_frequency{ Input::energy_recording_frequency };
    std::pair<int, Range> field_recording_frequency{ Input::field_recording_frequency };
    std::pair<int, Range> moment_recording_frequency{ Input::moment_recording_frequency };
    std::pair<int, Range> particle_recording_frequency{ Input::particle_recording_frequency };
    std::pair<int, Range> vhistogram_recording_frequency{ Input::vhistogram_recording_frequency };
    //
    ParamSet() = default;
    ParamSet(long subdomain_rank, Options const &opts);

private:
    // serializer
    //
    template <class... Ts, class Int, Int... Is>
    [[nodiscard]] static constexpr auto helper_cat(std::tuple<Ts...> const &t, std::integer_sequence<Int, Is...>) noexcept
    {
        return std::tuple_cat(serialize(std::get<Is>(t))...);
    }
    [[nodiscard]] friend constexpr auto serialize(ParamSet const &params) noexcept
    {
        auto const global = std::make_tuple(
            params.algorithm,
            (Debug::should_use_unified_snapshot ? params.number_of_distributed_particle_subdomain_clones : 0),
            params.particle_boundary_condition, params.should_randomize_gyrophase_of_reflecting_particles,
            params.phase_retardation.masking_inset, params.phase_retardation.masking_factor,
            params.amplitude_damping.masking_inset, params.amplitude_damping.masking_factor,
            params.c, params.O0, params.xi, params.Dx, params.Nx, params.dt, params.inner_Nt);
        auto const efluid = serialize(params.efluid_desc);
        auto const parts  = helper_cat(params.part_descs, part_indices{});
        auto const colds  = helper_cat(params.cold_descs, cold_indices{});
        auto const srcs   = helper_cat(params.source_descs, source_indices{});
        return std::tuple_cat(global, efluid, parts, colds, srcs);
    }

    // attribute export facility
    //
    friend auto operator<<(hdf5::Group &obj, ParamSet const &params) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, ParamSet const &params) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, ParamSet const &params) -> decltype(obj)
    {
        return std::move(obj << params);
    }
    friend auto operator<<(hdf5::Dataset &&obj, ParamSet const &params) -> decltype(obj)
    {
        return std::move(obj << params);
    }
};
HYBRID1D_END_NAMESPACE
