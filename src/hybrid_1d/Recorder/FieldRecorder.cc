/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "FieldRecorder.h"

#include <filesystem>
#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
auto FieldRecorder::filepath(std::string_view const &wd, long const step_count) const
{
    constexpr std::string_view prefix = "field";
    if (!is_world_master())
        throw std::domain_error{ __PRETTY_FUNCTION__ };

    auto const filename = std::string{ prefix } + "-" + std::to_string(step_count) + ".h5";
    return std::filesystem::path{ wd } / filename;
}

FieldRecorder::FieldRecorder(ParamSet const &params, parallel::mpi::Comm _subdomain_comm, parallel::mpi::Comm const &world_comm)
: Recorder{ params.field_recording_frequency, std::move(_subdomain_comm), world_comm }
{
}

void FieldRecorder::record(const Domain &domain, const long step_count)
{
    if (!should_record_at(step_count))
        return;

    if (is_world_master())
        record_master(domain, step_count);
    else
        record_worker(domain, step_count);
}

template <class Object>
decltype(auto) FieldRecorder::write_attr(Object &&obj, Domain const &domain, long const step)
{
    obj << domain.params;
    obj.attribute("step", hdf5::make_type(step), hdf5::Space::scalar()).write(step);

    auto const time = step * domain.params.dt;
    obj.attribute("time", hdf5::make_type(time), hdf5::Space::scalar()).write(time);

    return std::forward<Object>(obj);
}
template <class T>
auto FieldRecorder::write_data(std::vector<T> payload, hdf5::Group &root, char const *name)
{
    auto const [mspace, fspace] = get_space(payload);
    auto const type             = hdf5::make_type<Real>();
    auto       dset             = root.dataset(name, type, fspace);
    dset.write(fspace, payload.data(), type, mspace);
    return dset;
}

void FieldRecorder::record_master(const Domain &domain, const long step_count)
{
    auto const path = filepath(domain.params.working_directory, step_count);

    // create hdf file and root group
    auto root = hdf5::File(hdf5::File::trunc_tag{}, path.c_str())
                    .group("field", hdf5::PList::gapl(), hdf5::PList::gcpl());

    // attributes
    write_attr(root, domain, step_count);

    // datasets
    auto const writer = [](auto payload, auto &root, auto *name) {
        return write_data(std::move(payload), root, name);
    };
    auto const &comm = subdomain_comm;

    if (auto obj = comm.gather<1>(cart_to_mfa(domain.bfield), master)
                       .unpack(writer, root, "B"))
        write_attr(std::move(obj), domain, step_count) << domain.bfield;

    if (auto obj = comm.gather<1>(cart_to_mfa(domain.efield), master)
                       .unpack(writer, root, "E"))
        write_attr(std::move(obj), domain, step_count) << domain.efield;

    root.flush();
}
void FieldRecorder::record_worker(const Domain &domain, const long)
{
    auto const &comm = subdomain_comm;

    comm.gather<1>(cart_to_mfa(domain.bfield), master).unpack([](auto) {});
    comm.gather<1>(cart_to_mfa(domain.efield), master).unpack([](auto) {});
}

auto FieldRecorder::cart_to_mfa(BField const &bfield) -> std::vector<MFAVector>
{
    std::vector<MFAVector> result;
    result.reserve(bfield.size());
    auto const q1min = bfield.grid_subdomain_extent().min();
    for (long i = 0; i < bfield.size(); ++i) {
        result.push_back(bfield.geomtr.cart_to_mfa(bfield[i], CurviCoord{ i + q1min }));
    }
    return result;
}
auto FieldRecorder::cart_to_mfa(EField const &efield) -> std::vector<MFAVector>
{
    std::vector<MFAVector> result;
    result.reserve(efield.size());
    auto const q1min = efield.grid_subdomain_extent().min();
    for (long i = 0; i < efield.size(); ++i) {
        result.push_back(efield.geomtr.cart_to_mfa(efield[i], CurviCoord{ i + q1min }));
    }
    return result;
}
HYBRID1D_END_NAMESPACE
