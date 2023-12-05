/*
 * Copyright (c) 2020-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "VHistogramRecorder.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
namespace {
constexpr LocalSample &operator+=(LocalSample &lhs, LocalSample const &rhs) noexcept
{
    lhs.marker += rhs.marker;
    lhs.weight += rhs.weight;
    lhs.real_f += rhs.real_f;
    return lhs;
}
constexpr MFAVector &assign(MFAVector &lhs, LocalSample const &rhs) noexcept
{
    return lhs = MFAVector(rhs.marker, rhs.weight, rhs.real_f);
}
[[nodiscard]] constexpr auto operator+(std::pair<long, long> pair, long const val) noexcept
{
    return std::make_pair(pair.first + val, pair.second + val);
}
} // namespace

auto VHistogramRecorder::filepath(std::string_view const &wd, long const step_count) const
{
    constexpr std::string_view prefix = "vhist2d";
    if (!is_world_master())
        throw std::domain_error{ __PRETTY_FUNCTION__ };

    auto const filename = std::string{ prefix } + "-" + std::to_string(step_count) + ".h5";
    return std::filesystem::path{ wd } / filename;
}

VHistogramRecorder::VHistogramRecorder(ParamSet const &params, parallel::mpi::Comm _subdomain_comm, parallel::mpi::Comm const &world_comm)
: Recorder{ params.vhistogram_recording_frequency, std::move(_subdomain_comm), world_comm }
{
    if (!(this->world_comm = world_comm.duplicated())->operator bool())
        throw std::domain_error{ __PRETTY_FUNCTION__ };
}

void VHistogramRecorder::record(const Domain &domain, const long step_count)
{
    if (!should_record_at(step_count))
        return;

    if (is_world_master())
        record_master(domain, step_count);
    else
        record_worker(domain, step_count);
}

class VHistogramRecorder::Indexer {
    // preconditions:
    // 1. length of span is positive
    // 2. dim is positive
    //
    Range    v1span;
    Range    v2span;
    unsigned v1dim;
    unsigned v2dim;

public:
    constexpr Indexer(Range const &v1span, unsigned const &v1dim, Range const &v2span, unsigned const &v2dim) noexcept
    : v1span{ v1span }, v2span{ v2span }, v1dim{ v1dim }, v2dim{ v2dim }
    {
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return (v1dim > 0) && (v2dim > 0);
    }

    static constexpr index_pair_t npos{
        std::numeric_limits<long>::max(),
        std::numeric_limits<long>::max(),
    };

    [[nodiscard]] auto operator()(Real const v1, Real const v2) const noexcept
    {
        // zero-based indexing
        //
        index_pair_t const idx = {
            static_cast<index_pair_t::first_type>(std::floor((v1 - v1span.min()) * v1dim / v1span.len)),
            static_cast<index_pair_t::second_type>(std::floor((v2 - v2span.min()) * v2dim / v2span.len)),
        };

        if (within(idx, std::make_pair(0, 0), std::make_pair(v1dim, v2dim), std::make_index_sequence<std::tuple_size_v<index_pair_t>>{}))
            return idx;

        return npos;
    }

private:
    template <std::size_t... I>
    [[nodiscard]] static bool within(index_pair_t const &idx, index_pair_t const &min, index_pair_t const &max, std::index_sequence<I...>) noexcept
    {
        return (... && (std::get<I>(idx) >= std::get<I>(min))) && (... && (std::get<I>(idx) < std::get<I>(max)));
    }
};

template <class Object>
decltype(auto) VHistogramRecorder::write_attr(Object &&obj, Domain const &domain, long const step)
{
    obj << domain.params;
    obj.attribute("step", hdf5::make_type(step), hdf5::Space::scalar()).write(step);

    auto const time = step * domain.params.dt;
    obj.attribute("time", hdf5::make_type(time), hdf5::Space::scalar()).write(time);

    obj.attribute("vhistogram_recording_domain_extent",
                  hdf5::make_type(domain.params.vhistogram_recording_domain_extent.minmax()),
                  hdf5::Space::scalar())
        .write(domain.params.vhistogram_recording_domain_extent.minmax());

    return std::forward<Object>(obj);
}
void VHistogramRecorder::write_data(hdf5::Group &root, global_vhist_t vhist)
{
    using hdf5::make_type;
    using hdf5::Space;
    using Value  = decltype(vhist)::value_type;
    using Index  = decltype(vhist)::key_type;
    using Mapped = decltype(vhist)::mapped_type;
    {
        std::vector<Index> payload(vhist.size());
        std::transform(begin(vhist), end(vhist), begin(payload), std::mem_fn(&Value::first));

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<std::tuple_element_t<0, Index>>();
        auto       dset             = root.dataset("idx", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
    {
        std::vector<Mapped> payload(vhist.size());
        std::transform(begin(vhist), end(vhist), begin(payload), std::mem_fn(&Value::second));

        auto const [mspace, fspace] = get_space(payload);
        auto const type             = hdf5::make_type<Real>();
        auto       dset             = root.dataset("psd", type, fspace);
        dset.write(fspace, payload.data(), type, mspace);
    }
}

void VHistogramRecorder::record_master(const Domain &domain, long step_count)
{
    // create hdf file and root group
    auto const  path = filepath(domain.params.working_directory, step_count);
    hdf5::Group root;

    std::vector<unsigned> spids;
    for (unsigned s = 0; s < domain.part_species.size(); ++s) {
        PartSpecies const &sp       = domain.part_species[s];
        auto const [v1span, v1divs] = Input::v1hist_specs.at(s);
        auto const [v2span, v2divs] = Input::v2hist_specs.at(s);
        Indexer const idxer{ v1span, v1divs, v2span, v2divs };
        if (!idxer)
            continue;

        if (v1span.len <= 0 || v2span.len <= 0)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid vspan extent: " + std::to_string(s) + "th species" };

        spids.push_back(s);
        if (!root) {
            root = hdf5::File(hdf5::File::trunc_tag{}, path.c_str())
                       .group("vhist2d", hdf5::PList::gapl(), hdf5::PList::gcpl());
            write_attr(root, domain, step_count);
        }

        // create species group
        auto parent = [&root, name = std::to_string(s)] {
            return root.group(name.c_str(), hdf5::PList::gapl(), hdf5::PList::gcpl());
        }();
        write_attr(parent, domain, step_count) << sp;
        {
            auto const v1lim = std::make_pair(v1span.min(), v1span.max());
            parent.attribute("v1lim", hdf5::make_type(v1lim), hdf5::Space::scalar())
                .write(v1lim);
            auto const v2lim = std::make_pair(v2span.min(), v2span.max());
            parent.attribute("v2lim", hdf5::make_type(v2lim), hdf5::Space::scalar())
                .write(v2lim);
            auto const vdims = std::make_pair(v1divs, v2divs);
            parent.attribute("vdims", hdf5::make_type(vdims), hdf5::Space::scalar())
                .write(vdims);
        }
        // velocity histogram
        write_data(parent, histogram(sp, idxer));
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
void VHistogramRecorder::record_worker(const Domain &domain, long const)
{
    for (unsigned s = 0; s < domain.part_species.size(); ++s) {
        PartSpecies const &sp       = domain.part_species[s];
        auto const [v1span, v1divs] = Input::v1hist_specs.at(s);
        auto const [v2span, v2divs] = Input::v2hist_specs.at(s);
        Indexer const idxer{ v1span, v1divs, v2span, v2divs };
        if (!idxer)
            continue;

        histogram(sp, idxer);
    }
}

auto VHistogramRecorder::histogram(PartSpecies const &sp, Indexer const &idxer) const -> global_vhist_t
{
    // select particles within the prescribed spatial extent
    std::vector<Particle const *> particle_bucket;
    particle_bucket.reserve(sp.bucket.size());
    for (auto const &ptl : sp.bucket) {
        if (sp.params.vhistogram_recording_domain_extent.is_member(ptl.pos.q1))
            particle_bucket.emplace_back(&ptl);
    }

    // counting
    //
    auto counted = global_counting(particle_bucket.size(), local_counting(sp, particle_bucket, idxer));

    // normalization & index shift
    // * one-based index
    // * drop npos index, which was a placeholder for out-of-range velocity
    global_vhist_t global_vhist;
    std::for_each(
        // it assumes all processes have at least one element in the map
        std::next(rbegin(counted.second)), rend(counted.second),
        [total_count = counted.first, &global_vhist](auto const &kv) {
            assign(global_vhist[kv.first + 1], kv.second) /= total_count;
        });

    return global_vhist;
}
auto VHistogramRecorder::global_counting(unsigned long local_count, local_vhist_t local_vhist) const
    -> std::pair<unsigned long /*total count*/, local_vhist_t>
{
    auto const &comm = world_comm;

    std::pair<unsigned long, local_vhist_t> counted{ 0, {} };

    auto tk1 = comm.ibsend(local_count, { master, tag });
    auto tk2 = comm.ibsend<local_vhist_t::value_type>(
        { std::make_move_iterator(local_vhist.begin()), std::make_move_iterator(local_vhist.end()) }, { master, tag });
    if (master == comm->rank()) {
        for (int rank = 0, size = comm.size(); rank < size; ++rank) {
            // count
            counted.first += *comm.recv<unsigned long>({ rank, tag });

            // histograms
            comm.recv<local_vhist_t::value_type>({}, { rank, tag })
                .unpack(
                    [](auto lwhist, local_vhist_t &vhist) {
                        std::for_each(
                            std::make_move_iterator(begin(lwhist)), std::make_move_iterator(end(lwhist)),
                            [&vhist](auto kv) {
                                vhist[kv.first] += std::move(kv).second;
                            });
                    },
                    counted.second);
        }
    } else {
        counted.first = 1;
        counted.second.try_emplace(Indexer::npos); // this is to make sure all non-master processes have at least one element
    }
    std::move(tk1).wait();
    std::move(tk2).wait();

    return counted;
}
auto VHistogramRecorder::local_counting(PartSpecies const &sp, std::vector<Particle const *> const &particle_bucket, Indexer const &idxer) -> local_vhist_t
{
    local_vhist_t local_vhist{};
    local_vhist.try_emplace(idxer.npos); // pre-allocate a slot for particles at out-of-range velocity
    auto const q1min = sp.grid_subdomain_extent().min();
    for (Particle const *ptr : particle_bucket) {
        auto const    &ptl = *ptr;
        Shape<1> const sh{ ptl.pos.q1 - q1min };

        auto V = sp.moment<1>().interp(sh);
        if (auto const n = Real{ sp.moment<0>().interp(sh) }; n < 1e-15) {
            V *= 0;
        } else {
            V /= n;
        }
        auto const &vel = sp.geomtr.cart_to_mfa(ptl.vel - V, ptl.pos);
        auto const &key = idxer(vel.x, std::sqrt(vel.y * vel.y + vel.z * vel.z));
        local_vhist[key] += LocalSample{ 1U, ptl.psd.weight, ptl.psd.real_f / ptl.psd.marker };
    }
    return local_vhist;
}
HYBRID1D_END_NAMESPACE
