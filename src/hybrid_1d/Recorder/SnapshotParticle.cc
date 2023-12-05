/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "SnapshotParticle.h"
#include "SnapshotHash.h"

#include <filesystem>
#include <stdexcept>
#include <vector>

HYBRID1D_BEGIN_NAMESPACE
SnapshotParticle::SnapshotParticle(parallel::mpi::Comm _world, ParamSet const &params)
: comm{ std::move(_world) }
, signature{ Hash{ serialize(params) }() }
, wd{ params.working_directory }
{
    if (!comm->operator bool())
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid mpi::Comm" };

    // method dispatch
    //
    if (is_master()) {
        save = &SnapshotParticle::save_master;
        load = &SnapshotParticle::load_master;
    } else {
        save = &SnapshotParticle::save_worker;
        load = &SnapshotParticle::load_worker;
    }
}

auto SnapshotParticle::filepath() const
{
    constexpr std::string_view basename  = "snapshot_particle";
    constexpr std::string_view extension = ".h5";

    auto path = std::filesystem::path{ wd } / basename;
    path.replace_extension(extension);
    return path;
}

void SnapshotParticle::save_helper(hdf5::Group &root, PartSpecies const &sp, std::string const &basename) const
{
    // create dataset
    auto dset = [&root, &basename] {
        auto dcpl = hdf5::PList::dcpl();
        dcpl.set_chunk(chunk_size);
        auto const type   = hdf5::make_type<Particle>();
        auto const fspace = hdf5::Space::simple(0U, H5S_UNLIMITED);
        return root.dataset(basename.c_str(), type, fspace, hdf5::PList::dapl(), dcpl);
    }() << sp;

    // write particles
    auto start = 0ULL;
    for (int rank = 0, size = comm.size(); rank < size; ++rank) {
        if (master != rank)
            comm.ibsend<long>(master, { rank, tag2 }).wait(); // ping
        std::vector<Particle> const payload = *comm.recv<Particle>({}, { rank, tag });

        auto const count = payload.size();
        dset.set_extent(start + count);

        auto fspace = dset.space();
        fspace.select(H5S_SELECT_SET, start, count);

        auto mspace = hdf5::Space::simple(count);
        mspace.select_all();

        dset.write(fspace, payload.data(), mspace);
        start += count;
    }
}
void SnapshotParticle::save_master(Domain const &domain, long const step_count) const &
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

    // particles
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto const &sp    = domain.part_species.at(i);
        auto const  gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto        group = root.group(gname.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl()) << sp;

        auto tk = comm.ibsend(sp.dump_ptls(), { master, tag });
        save_helper(group, sp, "particle");
        std::move(tk).wait();
    }

    root.flush();
}
void SnapshotParticle::save_worker(Domain const &domain, long) const &
{
    // particles
    for (PartSpecies const &sp : domain.part_species) {
        if (master != comm.recv<long>({ master, tag2 })) // pong
            throw std::domain_error{ std::string{ __PRETTY_FUNCTION__ } + " - handshake failed" };
        comm.ibsend(sp.dump_ptls(), { master, tag }).wait();
    }
}

class SnapshotParticle::LoaderPredicate {
    long      number_of_particles_loaded{};
    int const rank;
    int const size;

public:
    LoaderPredicate(LoaderPredicate const &) = delete;
    LoaderPredicate &operator=(LoaderPredicate const &) = delete;
    explicit LoaderPredicate(RankSize const &rank_size) noexcept
    : rank{ rank_size.rank }, size{ rank_size.size } {}
    [[nodiscard]] bool operator()(Particle const &) &noexcept
    {
        return rank == number_of_particles_loaded++ % size;
    }
};
void SnapshotParticle::load_helper(hdf5::Group const &root, PartSpecies &sp, std::string const &basename, LoaderPredicate &pred) const
{
    // open dataset
    auto const dset   = root.dataset(basename.c_str());
    auto       fspace = dset.space();
    if (!fspace.is_simple() || fspace.rank() != 1)
        throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - unexpected dataspace of " + basename };

    // distributor
    auto const distributor = [&](unsigned long const start, unsigned long const count) {
        // read data
        fspace.select(H5S_SELECT_SET, start, count);
        auto mspace = hdf5::Space::simple(count);
        mspace.select_all();
        std::vector<Particle> payload(count);
        dset.read(fspace, payload.data(), mspace);

        // distribute
        auto tk = comm.ibsend(std::move(payload), { 0, tag });
        distribute_particles(sp, pred);
        std::move(tk).wait();
    };
    //
    auto const total_count = fspace.simple_extent().first.at(0);
    auto const n_chunks    = total_count / chunk_size;
    for (auto i = 0UL; i < n_chunks; ++i) {
        distributor(i * chunk_size, chunk_size);
    }
    // leftover
    distributor(n_chunks * chunk_size, total_count % chunk_size); // the fact that count < chunk_size signals no more data
}
auto SnapshotParticle::distribute_particles(PartSpecies &sp, LoaderPredicate &pred) const -> unsigned long
{
    auto const rank = comm->rank();
    auto const prev = rank == 0 ? master : rank - 1;
    auto const next = rank == comm.size() - 1 ? parallel::mpi::Rank::null() : rank + 1;

    std::vector<Particle> payload = *comm.recv<Particle>({}, { prev, tag });
    unsigned long const   count   = payload.size();
    sp.load_ptls(payload, pred);
    comm.ibsend(std::move(payload), { next, tag }).wait();
    return count;
}
long SnapshotParticle::load_master(Domain &domain, RankSize const &rank_size) const &
{
    // open hdf5 file and root group
    hdf5::Group root = hdf5::File(hdf5::File::rdonly_tag{}, filepath().c_str())
                           .group("hybrid_1d");

    // verify signature
    if (signature != root.attribute("signature").read<decltype(signature)>())
        throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - signature verification failed" };

    // particles
    for (unsigned i = 0; i < domain.part_species.size(); ++i) {
        auto      &sp    = domain.part_species.at(i);
        auto const gname = std::string{ "part_species" } + '@' + std::to_string(i);
        auto const group = root.group(gname.c_str());
        sp.bucket.clear();
        LoaderPredicate pred{ rank_size };
        load_helper(group, sp, "particle", pred);
    }

    // step count
    auto const step_count = root.attribute("step_count").read<long>();
    return comm.bcast<long>(step_count, master).unpack([step_count](auto) {
        // ignoring the arg is deliberate because bcasting to root itself does not work for some MPI impls
        return step_count;
    });
}
long SnapshotParticle::load_worker(Domain &domain, RankSize const &rank_size) const &
{
    // particles
    for (PartSpecies &sp : domain.part_species) {
        sp.bucket.clear();
        LoaderPredicate pred{ rank_size };
        while (distribute_particles(sp, pred) == chunk_size) {}
    }

    // step count
    return comm.bcast<long>(long{}, master);
}
HYBRID1D_END_NAMESPACE
