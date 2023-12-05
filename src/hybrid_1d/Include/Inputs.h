/*
 * Copyright (c) 2019-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

/// Simulation input parameters
/// \details
/// Modify the variables under the `Input' struct.
///
/// Consult/search headers under libPIC for unknown types/symbols.
///
struct Input {
    //
    // MARK:- Environment
    //

    /// Number of ghost cells
    ///
    /// It must be greater than 0.
    ///
    static constexpr unsigned number_of_ghost_cells = 3;

    /// number of subdomains for domain decomposition (positive integer)
    ///
    /// Nx must be divisible by this number
    ///
    static constexpr unsigned number_of_subdomains = 4;

    /// number of subdomain clones on which evenly divided particles are assigned and updated (positive integer)
    ///
    static constexpr unsigned number_of_distributed_particle_subdomain_clones = 3;

    /// number of worker threads to spawn for parallelization
    ///
    /// value `0' means serial update; value `n' means parallelization using n + 1 threads
    /// n + 1 must be divisible by number_of_subdomains * number_of_distributed_particle_subdomain_clones
    ///
    static constexpr unsigned number_of_worker_threads
        = 1 * number_of_subdomains * number_of_distributed_particle_subdomain_clones - 1;

    /// electric field extrapolation method
    ///
    static constexpr Algorithm algorithm = CAMCL;

    /// number of subscyles for magnetic field update; applied only for CAM-CL algorithm
    ///
    static constexpr unsigned n_subcycles = 10;

    /// particle boundary condition
    ///
    static constexpr BC particle_boundary_condition = BC::periodic;

    /// if set, randomize the gyro-phase of reflected particles
    ///
    static constexpr bool should_randomize_gyrophase_of_reflecting_particles = true;

    /// wave masking function
    ///
    /// the first argument is masking inset, i.e., the number of grid points through which waves are gradually damped
    /// the second argument is the masking coefficients, zero being no masking at all and one being 0 to 100% masking within the masking inset
    ///
    /// see `docs/boundary_condition.nb` for how the phase retardation and amplitude damping work
    ///
    static constexpr MaskingFunction phase_retardation{};
    static constexpr MaskingFunction amplitude_damping{};

    //
    // MARK: Global parameters
    //

    /// light speed
    ///
    static constexpr Real c = 214.243;

    /// magnitude of equatorial background magnetic field
    ///
    static constexpr Real O0 = 1;

    /// inhomogeneity parameter, ξ
    /// the field variation at the central field line is given by B/B_eq = 1 + (ξ*x)^2,
    /// where x is the coordinate along the axis of mirror field symmetry
    ///
    static constexpr Real xi = 0;

    /// simulation grid size at the equator
    ///
    static constexpr Real Dx = 0.2;

    /// number of grid points
    ///
    static constexpr unsigned Nx = 480;

    /// time step size
    ///
    static constexpr Real dt = 0.01;

    /// number of time steps for inner loop
    /// total time step Nt = inner_Nt * outer_Nt
    /// simulation time t = dt*Nt
    ///
    static constexpr unsigned inner_Nt = 15;

    /// number of time steps for outer loop
    /// total time step Nt = inner_Nt * outer_Nt
    /// simulation time t = dt*Nt
    ///
    /// this option is configurable through the commandline option, e.g., "--outer_Nt=100"
    ///
    static constexpr unsigned outer_Nt = 100;

    //
    // MARK: Plasma Species Descriptions
    //

    /// charge-neutralizing electron fluid description
    ///
    static constexpr auto efluid_desc = eFluidDesc({ -1836, 9180.01 }, 0, isothermal);

    /// kinetic plasma descriptors
    ///
    static constexpr auto part_descs = std::make_tuple(
        BiMaxPlasmaDesc({ { 1, c, 2 }, 1, TSC, full_f }, 1, 3),
        LossconePlasmaDesc({}, BiMaxPlasmaDesc{ { { 1, c, 2 }, 1000, TSC, full_f }, 1, 1 }));

    /// cold fluid plasma descriptors
    ///
    static constexpr auto cold_descs = std::make_tuple(ColdPlasmaDesc({ 1, 207.716, 2 }));

    /// external source descriptors
    ///
    static constexpr auto source_descs = std::make_tuple(
        ExternalSourceDesc<1>{ { 1, { 10, 100 }, 3, 2 }, { ComplexVector{ 1, 1i, 0 } }, { CurviCoord{ Nx + 1 } } });

    //
    // MARK: Data Recording
    //

    /// a top-level directory to which outputs will be saved
    ///
    /// this option is configurable through the commandline option, e.g., "--wd ./data"
    ///
    static constexpr std::string_view working_directory = "./data";

    /// a pair of
    ///
    /// - field and particle energy density recording frequency; in units of inner_Nt
    /// - recording start time and recording duration; in units of simulation time
    ///
    /// passing zero to the recording frequency means no recording
    ///
    /// the recording frequency option is configurable through the commandline option, e.g., "--energy_recording_frequency=10"
    ///
    static constexpr std::pair<unsigned, Range> energy_recording_frequency = { 1, {} };

    /// a pair of
    ///
    /// - electric and magnetic field recording frequency; in units of inner_Nt
    /// - recording start time and recording duration; in units of simulation time
    ///
    /// passing zero to the recording frequency means no recording
    ///
    /// the recording frequency option is configurable through the commandline option, e.g., "--field_recording_frequency=10"
    ///
    static constexpr std::pair<unsigned, Range> field_recording_frequency = { 5, {} };

    /// a pair of
    ///
    /// - ion species moment recording frequency; in units of inner_Nt
    /// - recording start time and recording duration; in units of simulation time
    ///
    /// passing zero to the recording frequency means no recording
    ///
    /// the recording frequency option is configurable through the commandline option, e.g., "--moment_recording_frequency=10"
    ///
    static constexpr std::pair<unsigned, Range> moment_recording_frequency = { 10, {} };

    /// a pair of
    ///
    /// - simulation particle recording frequency; in units of inner_Nt
    /// - recording start time and recording duration; in units of simulation time
    ///
    /// passing zero to the recording frequency means no recording
    ///
    /// the recording frequency option is configurable through the commandline option, e.g., "--particle_recording_frequency=10"
    ///
    static constexpr std::pair<unsigned, Range> particle_recording_frequency = { 20, {} };

    /// the spatial extent within which particles should be sampled
    ///
    /// note that the Range type is initialized with an OFFSET (or location) and LENGTH
    /// to choose the whole domain, set this to { -0.5 * Nx, Nx }
    ///
    /// NOTE: since the type of Nx is unsigned, -Nx will be a positive number!
    ///       be sure to cast to a double!
    ///
    static constexpr Range particle_recording_domain_extent
        = Range{ -0.5 * Real{ Nx }, Nx };

    /// maximum number of particles to dump
    ///
    static constexpr std::array<unsigned, std::tuple_size_v<decltype(part_descs)>> Ndumps
        = { 1000000, 1000000 };

    /// a pair of
    ///
    /// - velocity histogram recording frequency; in units of inner_Nt
    /// - recording start time and recording duration; in units of simulation time
    ///
    /// passing zero to the recording frequency means no recording
    ///
    /// the recording frequency option is configurable through the commandline option, e.g., "--vhistogram_recording_frequency=10"
    ///
    static constexpr std::pair<unsigned, Range> vhistogram_recording_frequency = { 5, {} };

    /// the spatial extent within which particles should be sampled
    ///
    /// note that the Range type is initialized with an OFFSET (or location) and LENGTH
    /// to choose the whole domain, set this to { -0.5 * Nx, Nx }
    ///
    /// NOTE: since the type of Nx is unsigned, -Nx will be a positive number!
    ///       be sure to cast to a double!
    ///
    static constexpr Range vhistogram_recording_domain_extent
        = Range{ -0.5 * Real{ Nx }, Nx };

    /// per-species gyro-averaged velocity space specification used for sampling velocity histogram
    ///
    /// the parallel (v1) and perpendicular (v2) velocity specs are described by
    /// the range of the velocity space extent and the number of velocity bins
    ///
    /// note that the Range type is initialized with an OFFSET (or location) and LENGTH
    ///
    /// recording histograms corresponding to specifications with the bin count being 0 will be
    /// skipped over
    ///
    static constexpr std::array<std::pair<Range, unsigned>, std::tuple_size_v<decltype(part_descs)>>
        v1hist_specs = { std::make_pair(Range{ -2, 5 }, 17) };
    static constexpr std::array<std::pair<Range, unsigned>, std::tuple_size_v<decltype(part_descs)>>
        v2hist_specs = { std::make_pair(Range{ 0, 1 }, 10) };
};

/// debugging options
///
namespace Debug {
constexpr bool zero_out_electromagnetic_field = false;
constexpr Real initial_bfield_noise_amplitude = 0e0;
constexpr bool should_use_unified_snapshot    = false;
} // namespace Debug
