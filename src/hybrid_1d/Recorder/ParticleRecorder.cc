/*
 * Copyright (c) 2019-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "ParticleRecorder.h"
#include "MomentRecorder.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
auto ParticleRecorder::filepath(std::string_view const &wd, long const step_count) const
{
    constexpr std::string_view prefix = "particle";
    if (!is_world_master())
        throw std::domain_error{ __PRETTY_FUNCTION__ };

    auto const filename = std::string{ prefix } + "-" + std::to_string(step_count) + ".h5";
    return std::filesystem::path{ wd } / filename;
}

ParticleRecorder::ParticleRecorder(ParamSet const &params, parallel::mpi::Comm _subdomain_comm, parallel::mpi::Comm const &world_comm)
: Recorder{ params.particle_recording_frequency, std::move(_subdomain_comm), world_comm }
, urbg{ 4988475U }
{
    if (!(this->world_comm = world_comm.duplicated())->operator bool())
        throw std::domain_error{ __PRETTY_FUNCTION__ };
}

void ParticleRecorder::record(const Domain &domain, const long step_count)
{
    if (!should_record_at(step_count))
        return;

    if (is_world_master())
        record_master(domain, step_count);
    else
        record_worker(domain, step_count);
}

template <class Object>
decltype(auto) ParticleRecorder::write_attr(Object &&obj, Domain const &domain, long const step)
{
    obj << domain.params;
    obj.attribute("step", hdf5::make_type(step), hdf5::Space::scalar()).write(step);

    auto const time = step * domain.params.dt;
    obj.attribute("time", hdf5::make_type(time), hdf5::Space::scalar()).write(time);

    obj.attribute("particle_recording_domain_extent",
                  hdf5::make_type(domain.params.particle_recording_domain_extent.minmax()),
                  hdf5::Space::scalar())
        .write(domain.params.particle_recording_domain_extent.minmax());

    return std::forward<Object>(obj);
}
template <class T>
auto ParticleRecorder::write_data(std::vector<T> payload, hdf5::Group &root, char const *name)
{
    auto const [mspace, fspace] = get_space(payload);
    auto const type             = hdf5::make_type<Real>();
    auto       dset             = root.dataset(name, type, fspace);
    dset.write(fspace, payload.data(), type, mspace);
    return dset;
}
void ParticleRecorder::write_data(std::vector<Particle> ptls, hdf5::Group &root, Geometry const &geomtr)
{
    // sort by particle id
    std::sort(begin(ptls), end(ptls), [](Particle const &a, Particle const &b) {
        return std::less{}(a.id, b.id);
    });

    using hdf5::make_type;
    using hdf5::Space;
    {
        std::vector<MFAVector> payload(ptls.size());
        std::transform(cbegin(ptls), cend(ptls), begin(payload), [&geomtr](Particle const &ptl) {
            return geomtr.cart_to_mfa(ptl.vel, ptl.pos);
        });

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<Real>();
        auto       dset             = root.dataset("vel", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        std::vector<CurviCoord> payload(ptls.size());
        std::transform(cbegin(ptls), cend(ptls), begin(payload), std::mem_fn(&Particle::pos));

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<Real>();
        auto       dset             = root.dataset("pos", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        std::vector<Particle::PSD> payload(ptls.size());
        std::transform(cbegin(ptls), cend(ptls), begin(payload), std::mem_fn(&Particle::psd));

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<Real>();
        auto       dset             = root.dataset("psd", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        std::vector<long> payload(ptls.size());
        std::transform(cbegin(ptls), cend(ptls), begin(payload), std::mem_fn(&Particle::id));

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<long>();
        auto       dset             = root.dataset("id", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
}

void ParticleRecorder::record_master(const Domain &domain, long const step_count)
{
    // create hdf file and root group
    auto const  path = filepath(domain.params.working_directory, step_count);
    hdf5::Group root;

    std::vector<unsigned> spids;
    for (unsigned s = 0; s < domain.part_species.size(); ++s) {
        PartSpecies const &sp    = domain.part_species[s];
        auto const         Ndump = Input::Ndumps.at(s);
        if (!Ndump)
            continue;

        spids.push_back(s);
        if (!root) {
            root = hdf5::File(hdf5::File::trunc_tag{}, path.c_str())
                       .group("particle", hdf5::PList::gapl(), hdf5::PList::gcpl());
            write_attr(root, domain, step_count);
        }

        // create species group
        auto parent = [&root, name = std::to_string(s)] {
            return root.group(name.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl());
        }();
        write_attr(parent, domain, step_count) << sp;
        parent.attribute("Ndump", hdf5::make_type(Ndump), hdf5::Space::scalar())
            .write(Ndump);

        // moments
        auto const writer = [](auto payload, auto &root, auto *name) {
            return write_data(std::move(payload), root, name);
        };
        if (auto const &comm = subdomain_comm; comm->operator bool()) {
            comm.gather<0>({ sp.moment<0>().begin(), sp.moment<0>().end() }, master)
                .unpack(writer, parent, "n");
            comm.gather<1>(MomentRecorder::cart_to_mfa(sp.moment<1>(), sp), master)
                .unpack(writer, parent, "nV");
            comm.gather<2>(MomentRecorder::cart_to_mfa(sp.moment<2>(), sp), master)
                .unpack(writer, parent, "nvv");
        }

        // particles
        write_data(collect_particles(sample(sp, Ndump)), parent, sp.geomtr);
    }

    // save species id's
    if (root) {
        auto space = hdf5::Space::simple(spids.size());
        auto dset  = root.dataset("spids", hdf5::make_type<decltype(spids)::value_type>(), space);
        space.select_all();
        dset.write(space, spids.data(), space);
        root.flush();
    }
}
void ParticleRecorder::record_worker(const Domain &domain, long const)
{
    for (unsigned s = 0; s < domain.part_species.size(); ++s) {
        PartSpecies const &sp    = domain.part_species[s];
        auto const         Ndump = Input::Ndumps.at(s);
        if (!Ndump)
            continue;

        // moments
        if (auto const &comm = subdomain_comm; comm->operator bool()) {
            comm.gather<0>({ sp.moment<0>().begin(), sp.moment<0>().end() }, master).unpack([](auto) {});
            comm.gather<1>(MomentRecorder::cart_to_mfa(sp.moment<1>(), sp), master).unpack([](auto) {});
            comm.gather<2>(MomentRecorder::cart_to_mfa(sp.moment<2>(), sp), master).unpack([](auto) {});
        }

        // particles
        collect_particles(sample(sp, Ndump));
    }
}
auto ParticleRecorder::collect_particles(std::vector<Particle> payload) -> std::vector<Particle>
{
    if (auto const &comm = world_comm; master == comm->rank()) {
        payload.reserve(payload.size() * unsigned(comm.size()));
        for (int rank = 0, size = comm.size(); rank < size; ++rank) {
            if (master == rank)
                continue;

            comm.issend<int>(master, { rank, tag }).wait();
            comm.recv<Particle>({}, { rank, tag })
                .unpack(
                    [](auto payload, std::vector<Particle> &buffer) {
                        buffer.insert(buffer.end(), std::make_move_iterator(begin(payload)), std::make_move_iterator(end(payload)));
                    },
                    payload);
        }
        return payload;
    } else {
        comm.recv<int>({ master, tag }).unpack([](auto) {});
        comm.ibsend(std::move(payload), { master, tag }).wait();
        return {};
    }
}

auto ParticleRecorder::sample(PartSpecies const &sp, unsigned long const max_count) -> std::vector<Particle>
{
    auto const &comm = world_comm;

    // select particles within the prescribed spatial extent
    std::vector<Particle const *> particle_bucket;
    particle_bucket.reserve(sp.bucket.size());
    for (auto const &ptl : sp.bucket) {
        if (sp.params.particle_recording_domain_extent.is_member(ptl.pos.q1))
            particle_bucket.emplace_back(&ptl);
    }

    // communicate the extent of indices of particles in this process
    auto const this_count   = particle_bucket.size();
    auto const index_extent = [&comm, this_count] {
        auto const self = comm->rank();
        auto const next = self == comm.size() - 1 ? parallel::mpi::Rank::null() : self + 1;
        if (self == 0) {
            unsigned long const offset = 0;
            comm.issend(offset + this_count, { next, tag }).wait();
            return std::make_pair(offset, offset + this_count);
        } else {
            unsigned long const offset = comm.recv<unsigned long>({ self - 1, tag });
            comm.issend(offset + this_count, { next, tag }).wait();
            return std::make_pair(offset, offset + this_count);
        }
    }();
    if (index_extent.first + this_count != index_extent.second)
        throw std::logic_error{ __PRETTY_FUNCTION__ };

    // shuffle particle indices and truncate at the max_count
    // NOTE: To mitigate OOM, shuffling is done on the root process and the result is broadcast.
    //       This is still not perfect, since every process tries to get hold of memory when
    //       the max_count gets large.
    //       I tried the same approach as the communication of index extent above, but creating
    //       indices and shuffling them is too slow when the process count gets too large.
    //       For now, this seems to be the happy medium.
    unsigned long const        total_count = comm.all_reduce(parallel::mpi::ReduceOp::plus(), this_count);
    std::vector<unsigned long> indices     = [this, &comm, max_count, total_count]() -> std::vector<unsigned long> {
        constexpr auto root = 0;
        if (comm->rank() == root) {
            std::vector<unsigned long> indices(total_count);
            std::iota(begin(indices), end(indices), 0U);
            std::shuffle(begin(indices), end(indices), urbg);
            if (max_count < indices.size()) {
                indices.resize(max_count);
                indices.shrink_to_fit();
            }
            (void)comm.bcast(indices.size(), root);
            return comm.bcast(std::move(indices), root);
        } else {
            std::vector<unsigned long> indices(comm.bcast(0UL, root));
            return comm.bcast(std::move(indices), root);
        }
    }();

    indices.erase(
        std::partition(
            begin(indices), end(indices), [index_extent](unsigned long const index) {
                return index >= index_extent.first && index < index_extent.second;
            }),
        end(indices));
    indices.shrink_to_fit();

    // sample particles
    std::vector<Particle> samples(indices.size());
    std::transform(
        begin(indices), end(indices), begin(samples),
        [&bucket = particle_bucket, offset = index_extent.first](unsigned long const index) {
            return *bucket.at(index - offset);
        });

    return samples;
}
HYBRID1D_END_NAMESPACE
