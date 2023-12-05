/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Snapshot.h"
#include "SnapshotHash.h"

#include <cstddef> // offsetof
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <type_traits>

HYBRID1D_BEGIN_NAMESPACE
Snapshot::Snapshot(parallel::mpi::Comm _comm, ParamSet const &params, long const subdomain_color)
: comm{ std::move(_comm) }
, signature{ Hash{ serialize(params) }() }
, wd{ params.working_directory }
{
    if (!comm->operator bool())
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid mpi::Comm" };

    if (subdomain_color >= 0)
        (filename_suffix = '_') += std::to_string(subdomain_color);

    // method dispatch
    //
    if (is_master()) {
        save = &Snapshot::save_master;
        load = &Snapshot::load_master;
    } else {
        save = &Snapshot::save_worker;
        load = &Snapshot::load_worker;
    }
}

auto Snapshot::filepath() const
{
    constexpr std::string_view basename  = "snapshot";
    constexpr std::string_view extension = ".h5";

    auto path = std::filesystem::path{ wd } / basename;
    path.concat(filename_suffix).replace_extension(extension);
    return path;
}

template <class T, long N>
auto Snapshot::save_helper(hdf5::Group &root, GridArray<T, N, Pad> const &grid, std::string const &basename) const -> hdf5::Dataset
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
void Snapshot::save_helper(hdf5::Group &root, PartSpecies const &sp) const
{
    // collect
    std::vector<Particle> payload;
    {
        long const Np = sp->Nc * sp.params.Nx / sp.params.number_of_distributed_particle_subdomain_clones;
        payload.reserve(static_cast<unsigned long>(Np));
    }
    auto tk = comm.ibsend(sp.dump_ptls(), { master, tag });
    for (int rank = 0, size = comm.size(); rank < size; ++rank) {
        comm.recv<Particle>({}, { rank, tag })
            .unpack(
                [](auto incoming, auto &payload) {
                    payload.insert(payload.end(), std::make_move_iterator(begin(incoming)), std::make_move_iterator(end(incoming)));
                },
                payload);
    }
    std::move(tk).wait();

    // export
    constexpr auto unit_size = sizeof(Real);
    static_assert(sizeof(Particle) % unit_size == 0);
    static_assert(alignof(Particle) == alignof(Real));
    auto mspace = hdf5::Space::simple({ payload.size(), sizeof(Particle) / unit_size });
    {
        auto const type = hdf5::make_type<Real>();
        using T         = std::decay_t<decltype(std::declval<Particle>().vel)>;

        hdf5::Extent const start = { 0U, offsetof(Particle, vel) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        auto fspace = hdf5::Space::simple(count);

        auto dset = root.dataset("vel", type, fspace, hdf5::PList::dapl(), hdf5::PList::dcpl()) << sp;
        fspace.select_all();
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        auto const type = hdf5::make_type<Real>();
        using T         = std::decay_t<decltype(std::declval<Particle>().pos)>;

        hdf5::Extent const start = { 0U, offsetof(Particle, pos) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        auto fspace = hdf5::Space::simple(count);

        auto dset = root.dataset("pos", type, fspace, hdf5::PList::dapl(), hdf5::PList::dcpl()) << sp;
        fspace.select_all();
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        auto const type = hdf5::make_type<Real>();
        using T         = std::decay_t<decltype(std::declval<Particle>().psd)>;

        hdf5::Extent const start = { 0U, offsetof(Particle, psd) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        auto fspace = hdf5::Space::simple(count);

        auto dset = root.dataset("psd", type, fspace, hdf5::PList::dapl(), hdf5::PList::dcpl()) << sp;
        fspace.select_all();
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        auto const type = hdf5::make_type<long>();
        using T         = std::decay_t<decltype(std::declval<Particle>().id)>;

        hdf5::Extent const start = { 0U, offsetof(Particle, id) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        auto fspace = hdf5::Space::simple(payload.size());

        auto dset = root.dataset("id", type, fspace, hdf5::PList::dapl(), hdf5::PList::dcpl()) << sp;
        fspace.select_all();
        dset.write(fspace, payload.data(), type, mspace);
    }
}
void Snapshot::save_master(Domain const &domain, long const step_count) const &
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

    // particles
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto const &sp    = domain.part_species.at(i);
        auto const  gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto        group = root.group(gname.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl()) << sp;
        save_helper(group, sp.equilibrium_mom0, "equilibrium_mom0") << sp;
        save_helper(group, sp.equilibrium_mom1, "equilibrium_mom1") << sp;
        save_helper(group, sp.equilibrium_mom2, "equilibrium_mom2") << sp;
        save_helper(group, sp);
    }

    // cold fluid
    for (unsigned i = 0; i < domain.cold_species.size(); ++i) {
        auto const &sp    = domain.cold_species.at(i);
        auto const  gname = std::string{ "cold_species" } + '@' + std::to_string(i);
        auto        group = root.group(gname.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl()) << sp;
        save_helper(group, sp.mom0_full, "mom0_full") << sp;
        save_helper(group, sp.mom1_full, "mom1_full") << sp;
    }

    root.flush();
}
void Snapshot::save_worker(Domain const &domain, long) const &
{
    // B & E
    comm.gather<1>(domain.bfield.begin(), domain.bfield.end(), nullptr, master);
    comm.gather<1>(domain.efield.begin(), domain.efield.end(), nullptr, master);

    // particles
    for (PartSpecies const &sp : domain.part_species) {
        comm.gather<0>(sp.equilibrium_mom0.begin(), sp.equilibrium_mom0.end(), nullptr, master);
        comm.gather<1>(sp.equilibrium_mom1.begin(), sp.equilibrium_mom1.end(), nullptr, master);
        comm.gather<2>(sp.equilibrium_mom2.begin(), sp.equilibrium_mom2.end(), nullptr, master);
        comm.ibsend(sp.dump_ptls(), { master, tag }).wait();
    }

    // cold fluid
    for (ColdSpecies const &sp : domain.cold_species) {
        comm.gather<0>(sp.mom0_full.begin(), sp.mom0_full.end(), nullptr, master);
        comm.gather<1>(sp.mom1_full.begin(), sp.mom1_full.end(), nullptr, master);
    }
}

template <class T, long N>
void Snapshot::load_helper(hdf5::Group const &root, GridArray<T, N, Pad> &grid, std::string const &basename) const
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
void Snapshot::load_helper(hdf5::Group const &root, PartSpecies &sp) const
{
    std::vector<Particle> payload;

    // get particle count
    {
        auto const extent = root.dataset("id").space().simple_extent().first;
        if (extent.rank() != 1)
            throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - expects 1D array in obtaining particle count" };
        payload.resize(extent[0]);
    }

    // import
    constexpr auto unit_size = sizeof(Real);
    static_assert(sizeof(Particle) % unit_size == 0);
    static_assert(alignof(Particle) == alignof(Real));
    auto mspace = hdf5::Space::simple({ payload.size(), sizeof(Particle) / unit_size });
    {
        auto const type   = hdf5::make_type<Real>();
        using T           = std::decay_t<decltype(std::declval<Particle>().vel)>;
        auto       dset   = root.dataset("vel");
        auto       fspace = dset.space();
        auto const extent = fspace.simple_extent().first;
        if (extent.rank() != 2 || extent[0] != payload.size() || extent[1] != sizeof(T) / unit_size)
            throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - incompatible extent : vel" };

        hdf5::Extent const start = { 0U, offsetof(Particle, vel) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        fspace.select_all();
        dset.read(fspace, payload.data(), type, mspace);
    }
    {
        auto const type   = hdf5::make_type<Real>();
        using T           = std::decay_t<decltype(std::declval<Particle>().pos)>;
        auto       dset   = root.dataset("pos");
        auto       fspace = dset.space();
        auto const extent = fspace.simple_extent().first;
        if (extent.rank() != 2 || extent[0] != payload.size() || extent[1] != sizeof(T) / unit_size)
            throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - incompatible extent : pos" };

        hdf5::Extent const start = { 0U, offsetof(Particle, pos) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        fspace.select_all();
        dset.read(fspace, payload.data(), type, mspace);
    }
    {
        auto const type   = hdf5::make_type<Real>();
        using T           = std::decay_t<decltype(std::declval<Particle>().psd)>;
        auto       dset   = root.dataset("psd");
        auto       fspace = dset.space();
        auto const extent = fspace.simple_extent().first;
        if (extent.rank() != 2 || extent[0] != payload.size() || extent[1] != sizeof(T) / unit_size)
            throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - incompatible extent : psd" };

        hdf5::Extent const start = { 0U, offsetof(Particle, psd) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        fspace.select_all();
        dset.read(fspace, payload.data(), type, mspace);
    }
    {
        auto const type   = hdf5::make_type<long>();
        using T           = std::decay_t<decltype(std::declval<Particle>().id)>;
        auto       dset   = root.dataset("id");
        auto       fspace = dset.space();
        auto const extent = fspace.simple_extent().first;
        if (extent.rank() != 1 || extent[0] != payload.size())
            throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - incompatible extent : id" };

        hdf5::Extent const start = { 0U, offsetof(Particle, id) / unit_size };
        hdf5::Extent const count = { payload.size(), sizeof(T) / unit_size };
        mspace.select(H5S_SELECT_SET, start, count);
        fspace.select_all();
        dset.read(fspace, payload.data(), type, mspace);
    }

    // distribute
    // FIXME: Currently, a whole chunk of particles read from the disk are passed over all subdomains one by one.
    //        This will consume twice as much memory as the payload size as the receiving buffer should allocate the same amount of memory.
    //        In addition, another problem occurs when passing data across the MPI boundary if the chunk size is too big (presumably larger than the `int` size).
    //        Obviously, this is not ideal and a better mechanism should be in place.
    auto tk = comm.ibsend(std::move(payload), { 0, tag2 });
    distribute_particles(sp);
    std::move(tk).wait();
}
void Snapshot::distribute_particles(PartSpecies &sp) const
{
    auto const rank = comm->rank();
    auto const prev = rank == 0 ? master : rank - 1;
    auto const next = rank == comm.size() - 1 ? parallel::mpi::Rank::null() : rank + 1;
    {
        std::vector<Particle> payload = *comm.recv<Particle>({}, { prev, tag2 });
        sp.load_ptls(payload, [](auto const &) {
            return true;
        });
        comm.ibsend(std::move(payload), { next, tag2 }).wait();
    }
}
long Snapshot::load_master(Domain &domain) const &
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

    // particles
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto      &sp    = domain.part_species.at(i);
        auto const gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto const group = root.group(gname.c_str());
        load_helper(group, sp.equilibrium_mom0, "equilibrium_mom0");
        load_helper(group, sp.equilibrium_mom1, "equilibrium_mom1");
        load_helper(group, sp.equilibrium_mom2, "equilibrium_mom2");
        sp.bucket.clear();
        load_helper(group, sp);
    }

    // cold fluid
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
long Snapshot::load_worker(Domain &domain) const &
{
    // B & E
    comm.scatter<1>(nullptr, domain.bfield.begin(), domain.bfield.end(), master);
    comm.scatter<1>(nullptr, domain.efield.begin(), domain.efield.end(), master);

    // particles
    for (PartSpecies &sp : domain.part_species) {
        comm.scatter<0>(nullptr, sp.equilibrium_mom0.begin(), sp.equilibrium_mom0.end(), master);
        comm.scatter<1>(nullptr, sp.equilibrium_mom1.begin(), sp.equilibrium_mom1.end(), master);
        comm.scatter<2>(nullptr, sp.equilibrium_mom2.begin(), sp.equilibrium_mom2.end(), master);
        sp.bucket.clear();
        distribute_particles(sp);
    }

    // cold fluid
    for (ColdSpecies &sp : domain.cold_species) {
        comm.scatter<0>(nullptr, sp.mom0_full.begin(), sp.mom0_full.end(), master);
        comm.scatter<1>(nullptr, sp.mom1_full.begin(), sp.mom1_full.end(), master);
    }

    // step count
    return comm.bcast<long>(long{}, master);
}
HYBRID1D_END_NAMESPACE
