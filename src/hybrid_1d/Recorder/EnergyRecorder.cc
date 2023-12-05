/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "EnergyRecorder.h"
#include <PIC/UTL/println.h>

#include <filesystem>
#include <functional>
#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
auto EnergyRecorder::filepath(std::string_view const &wd) const
{
    constexpr std::string_view filename = "energy.csv";
    if (is_world_master())
        return std::filesystem::path{ wd } / filename;
    return std::filesystem::path{ "/dev/null" };
}

EnergyRecorder::EnergyRecorder(ParamSet const &params, parallel::mpi::Comm _subdomain_comm, parallel::mpi::Comm const &world_comm)
: Recorder{ params.energy_recording_frequency, std::move(_subdomain_comm), world_comm }
{
    // open output stream
    //
    auto const path = filepath(params.working_directory);
    if (os.open(path, params.snapshot_load ? os.app : os.trunc); !os)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - open failed: " + path.string() };

    os.setf(os.scientific);
    os.precision(15);

    if (!params.snapshot_load) {
        // header lines
        //
        print(os, "step");   // integral step count
        print(os, ", time"); // simulation time
        //
        print(os, ", dB1^2/2, dB2^2/2, dB3^2/2"); // spatial average of fluctuating (without background) magnetic field energy density
        print(os, ", dE1^2/2, dE2^2/2, dE3^2/2"); // spatial average of fluctuating (without background) electric field energy density
        //
        for (unsigned i = 1; i <= ParamSet::part_indices::size(); ++i) {
            // spatial average of i'th species kinetic energy density in lab frame
            print(os, ", part_species(", i, ") KE1_lab", ", part_species(", i, ") KE2_lab", ", part_species(", i, ") KE3_lab");
            // spatial average of i'th species kinetic energy density in plasma frame
            print(os, ", part_species(", i, ") KE1_pla", ", part_species(", i, ") KE2_pla", ", part_species(", i, ") KE3_pla");
        }
        //
        for (unsigned i = 1; i <= ParamSet::cold_indices::size(); ++i) {
            // spatial average of i'th species kinetic energy density in lab frame
            print(os, ", cold_species(", i, ") KE1_lab", ", cold_species(", i, ") KE2_lab", ", cold_species(", i, ") KE3_lab");
            // spatial average of i'th species kinetic energy density in plasma frame
            print(os, ", cold_species(", i, ") KE1_pla", ", cold_species(", i, ") KE2_pla", ", cold_species(", i, ") KE3_pla");
        }
        //
        os << std::endl;
    }
}

void EnergyRecorder::record(const Domain &domain, const long step_count)
{
    if (!should_record_at(step_count))
        return;

    print(os, step_count, ", ", step_count * domain.params.dt);

    auto printer = [&os = this->os](MFAVector const &v) {
        print(os, ", ", v.x, ", ", v.y, ", ", v.z);
    };
    auto const &comm = subdomain_comm;

    printer(*comm.all_reduce(std::plus{}, dump(domain.bfield)));
    printer(*comm.all_reduce(std::plus{}, dump(domain.efield)));

    for (Species const &sp : domain.part_species) {
        auto const [KE_lab, KE_pla]
            = comm.all_reduce(std::plus{}, dump(sp)).unpack([](std::vector<MFAVector> vec) {
                  return std::make_pair(vec.at(0), vec.at(1));
              });
        printer(KE_lab); // in lab frame
        printer(KE_pla); // in plasma frame
    }

    for (Species const &sp : domain.cold_species) {
        auto const [KE_lab, KE_pla]
            = comm.all_reduce(std::plus{}, dump(sp)).unpack([](std::vector<MFAVector> vec) {
                  return std::make_pair(vec.at(0), vec.at(1));
              });
        printer(KE_lab); // in lab frame
        printer(KE_pla); // in plasma frame
    }

    os << std::endl;
}

auto EnergyRecorder::dump(BField const &bfield) -> MFAVector
{
    MFAVector  dB2O2{};
    auto const q1min = bfield.grid_subdomain_extent().min();
    for (long i = 0; i < bfield.size(); ++i) {
        auto const dB = bfield.geomtr.cart_to_mfa(bfield[i], CurviCoord{ i + q1min });
        dB2O2 += dB * dB;
    }
    dB2O2 /= 2 * Input::Nx;
    return dB2O2;
}
auto EnergyRecorder::dump(EField const &efield) -> MFAVector
{
    MFAVector  dE2O2{};
    auto const q1min = efield.grid_subdomain_extent().min();
    for (long i = 0; i < efield.size(); ++i) {
        auto const dE = efield.geomtr.cart_to_mfa(efield[i], CurviCoord{ i + q1min });
        dE2O2 += dE * dE;
    }
    dE2O2 /= 2 * Input::Nx;
    return dE2O2;
}
auto EnergyRecorder::dump(Species const &sp) -> std::vector<MFAVector>
{
    MFAVector  KE_lab{};
    MFAVector  KE_pla{};
    auto const q1min = sp.grid_subdomain_extent().min();
    for (long i = 0; i < sp.moment<0>().size(); ++i) {
        auto const pos = CurviCoord{ i + q1min };
        auto const n   = Real{ sp.moment<0>()[i] };
        if (constexpr auto zero = 1e-15; n < zero)
            continue;

        auto const nvv_lab = sp.geomtr.cart_to_mfa(sp.moment<2>()[i], pos);
        KE_lab += nvv_lab.lo();

        auto const nV = sp.geomtr.cart_to_mfa(sp.moment<1>()[i], pos);
        KE_pla += nvv_lab.lo() - nV * nV / n;
    }
    KE_lab *= sp.energy_density_conversion_factor() / (2 * Input::Nx);
    KE_pla *= sp.energy_density_conversion_factor() / (2 * Input::Nx);
    return { KE_lab, KE_pla };
}
HYBRID1D_END_NAMESPACE
