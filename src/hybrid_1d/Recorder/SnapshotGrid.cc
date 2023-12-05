/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "SnapshotGrid.h"
#include "SnapshotHash.h"

#include <filesystem>
#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
SnapshotGrid::SnapshotGrid(parallel::mpi::Comm _comm, ParamSet const &params)
: comm{ std::move(_comm) }
, signature{ Hash{ serialize(params) }() }
, wd{ params.working_directory }
{
    if (!comm->operator bool())
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid mpi::Comm" };

    // method dispatch
    //
    if (is_master()) {
        save = &SnapshotGrid::save_master;
        load = &SnapshotGrid::load_master;
    } else {
        save = &SnapshotGrid::save_worker;
        load = &SnapshotGrid::load_worker;
    }
}

auto SnapshotGrid::filepath() const
{
    constexpr std::string_view basename  = "snapshot_grid";
    constexpr std::string_view extension = ".h5";

    auto path = std::filesystem::path{ wd } / basename;
    path.replace_extension(extension);
    return path;
}

template <class T, long N>
auto SnapshotGrid::save_helper(hdf5::Group &root, GridArray<T, N, Pad> const &grid, std::string const &basename) const -> hdf5::Dataset
{
    static_assert(alignof(T) == alignof(Real), "memory and file type mis-alignment");
    static_assert(0 == sizeof(T) % sizeof(Real), "memory and file type size incompatible");
    constexpr auto len  = sizeof(T) / sizeof(Real);
    auto const     type = hdf5::make_type<Real>();

    std::vector<T> payload = *comm.gather<T>({ grid.begin(), grid.end() }, master);
    auto           mspace  = hdf5::Space::simple({ payload.size(), len });

    // fixed dataset
    auto fspace = hdf5::Space::simple({ payload.size(), len });
    auto dset   = root.dataset(basename.c_str(), type, fspace);

    // export
    mspace.select_all();
    fspace.select_all();
    dset.write(fspace, payload.data(), type, mspace);

    return dset;
}
void SnapshotGrid::save_master(Domain const &domain, long const step_count) const &
{
    // create hdf5 file and root group
    hdf5::Group root = hdf5::File(hdf5::File::trunc_tag{}, filepath().c_str())
                           .group("hybrid_1d", hdf5::PList::gapl(), hdf5::PList::gcpl());
    root << domain.params;

    // step_count & signature
    root.attribute("step_count", hdf5::make_type(step_count), hdf5::Space::scalar())
        .write(step_count);
    root.attribute("signature", hdf5::make_type(signature), hdf5::Space::scalar())
        .write(signature);

    // B & E
    save_helper(root, domain.bfield, "bfield") << domain.bfield;
    save_helper(root, domain.efield, "efield") << domain.efield;

    // particle equilibrium moments
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto const &sp    = domain.part_species.at(i);
        auto const  gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto        group = root.group(gname.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl()) << sp;
        save_helper(group, sp.equilibrium_mom0, "equilibrium_mom0") << sp;
        save_helper(group, sp.equilibrium_mom1, "equilibrium_mom1") << sp;
        save_helper(group, sp.equilibrium_mom2, "equilibrium_mom2") << sp;
    }

    // cold fluid moments
    for (unsigned i = 0; i < domain.cold_species.size(); ++i) {
        auto const &sp    = domain.cold_species.at(i);
        auto const  gname = std::string{ "cold_species" } + '@' + std::to_string(i);
        auto        group = root.group(gname.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl()) << sp;
        save_helper(group, sp.mom0_full, "mom0_full") << sp;
        save_helper(group, sp.mom1_full, "mom1_full") << sp;
    }

    root.flush();
}
void SnapshotGrid::save_worker(Domain const &domain, long) const &
{
    // B & E
    comm.gather<1>(domain.bfield.begin(), domain.bfield.end(), nullptr, master);
    comm.gather<1>(domain.efield.begin(), domain.efield.end(), nullptr, master);

    // particle equilibrium moments
    for (PartSpecies const &sp : domain.part_species) {
        comm.gather<0>(sp.equilibrium_mom0.begin(), sp.equilibrium_mom0.end(), nullptr, master);
        comm.gather<1>(sp.equilibrium_mom1.begin(), sp.equilibrium_mom1.end(), nullptr, master);
        comm.gather<2>(sp.equilibrium_mom2.begin(), sp.equilibrium_mom2.end(), nullptr, master);
    }

    // cold fluid moments
    for (ColdSpecies const &sp : domain.cold_species) {
        comm.gather<0>(sp.mom0_full.begin(), sp.mom0_full.end(), nullptr, master);
        comm.gather<1>(sp.mom1_full.begin(), sp.mom1_full.end(), nullptr, master);
    }
}

template <class T, long N>
void SnapshotGrid::load_helper(hdf5::Group const &root, GridArray<T, N, Pad> &grid, std::string const &basename) const
{
    static_assert(alignof(T) == alignof(Real), "memory and file type mis-alignment");
    static_assert(0 == sizeof(T) % sizeof(Real), "memory and file type size incompatible");
    constexpr auto len  = sizeof(T) / sizeof(Real);
    auto const     type = hdf5::make_type<Real>();

    std::vector<T> payload(static_cast<unsigned long>(grid.size() * comm.size()));
    auto           mspace = hdf5::Space::simple({ payload.size(), len });

    // open dataset
    auto       dset   = root.dataset(basename.c_str());
    auto       fspace = dset.space();
    auto const extent = fspace.simple_extent().first;
    if (extent.rank() != 2 || extent[0] != payload.size() || extent[1] != len)
        throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - incompatible extent : " + basename };

    // import
    mspace.select_all();
    fspace.select_all();
    dset.read(fspace, payload.data(), type, mspace);

    // distribute
    comm.scatter(payload.data(), grid.begin(), grid.end(), master);
}
long SnapshotGrid::load_master(Domain &domain) const &
{
    // open hdf5 file and root group
    hdf5::Group root = hdf5::File(hdf5::File::rdonly_tag{}, filepath().c_str())
                           .group("hybrid_1d");

    // verify signature
    if (signature != root.attribute("signature").read<decltype(signature)>())
        throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - signature verification failed" };

    // B & E
    load_helper(root, domain.bfield, "bfield");
    load_helper(root, domain.efield, "efield");

    // particle equilibrium moments
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto      &sp    = domain.part_species.at(i);
        auto const gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto const group = root.group(gname.c_str());
        load_helper(group, sp.equilibrium_mom0, "equilibrium_mom0");
        load_helper(group, sp.equilibrium_mom1, "equilibrium_mom1");
        load_helper(group, sp.equilibrium_mom2, "equilibrium_mom2");
    }

    // cold fluid moments
    for (unsigned i = 0; i < domain.cold_species.size(); ++i) {
        auto      &sp    = domain.cold_species.at(i);
        auto const gname = std::string{ "cold_species" } + '@' + std::to_string(i);
        auto const group = root.group(gname.c_str());
        load_helper(group, sp.mom0_full, "mom0_full");
        load_helper(group, sp.mom1_full, "mom1_full");
    }

    // step count
    auto const step_count = root.attribute("step_count").read<long>();
    return comm.bcast<long>(step_count, master).unpack([step_count](auto) {
        // ignoring the arg is deliberate because bcasting to root itself does not work for some MPI impls
        return step_count;
    });
}
long SnapshotGrid::load_worker(Domain &domain) const &
{
    // B & E
    comm.scatter<1>(nullptr, domain.bfield.begin(), domain.bfield.end(), master);
    comm.scatter<1>(nullptr, domain.efield.begin(), domain.efield.end(), master);

    // particle equilibrium moments
    for (PartSpecies &sp : domain.part_species) {
        comm.scatter<0>(nullptr, sp.equilibrium_mom0.begin(), sp.equilibrium_mom0.end(), master);
        comm.scatter<1>(nullptr, sp.equilibrium_mom1.begin(), sp.equilibrium_mom1.end(), master);
        comm.scatter<2>(nullptr, sp.equilibrium_mom2.begin(), sp.equilibrium_mom2.end(), master);
    }

    // cold fluid moments
    for (ColdSpecies &sp : domain.cold_species) {
        comm.scatter<0>(nullptr, sp.mom0_full.begin(), sp.mom0_full.end(), master);
        comm.scatter<1>(nullptr, sp.mom1_full.begin(), sp.mom1_full.end(), master);
    }

    // step count
    return comm.bcast<long>(long{}, master);
}
HYBRID1D_END_NAMESPACE
