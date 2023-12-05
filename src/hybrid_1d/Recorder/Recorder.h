/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Core/Domain.h"
#include <PIC/TypeMaps.h>

#include <HDF5Kit/HDF5Kit.h>
#include <ParallelKit/ParallelKit.h>
#include <type_traits>
#include <utility>
#include <vector>

HYBRID1D_BEGIN_NAMESPACE
class Recorder {
    long const  m_recording_frequency;
    Range const m_recording_temporal_extent;

public:
    virtual ~Recorder() = default;

    [[nodiscard]] virtual bool should_record_at(long step_count) const noexcept;
    virtual void               record(Domain const &domain, long step_count) = 0;

protected:
    Recorder(std::pair<int, Range> recording_frequency, parallel::mpi::Comm subdomain_comm, parallel::mpi::Comm const &world_comm);

    parallel::Communicator<Scalar, MFAVector, MFATensor> const subdomain_comm;
    bool                                                       m_is_world_master;

    static constexpr auto tag = parallel::mpi::Tag{ 875 };
    static constexpr auto master{ 0 };

    [[nodiscard]] bool is_world_master() const noexcept { return m_is_world_master; }

    // hdf5 space calculator
    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    [[nodiscard]] static auto get_space(std::vector<T> const &payload)
    {
        auto mspace = hdf5::Space::simple(payload.size());
        mspace.select_all();
        auto fspace = hdf5::Space::simple(payload.size());
        fspace.select_all();
        return std::make_pair(std::move(mspace), std::move(fspace));
    }
    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    [[nodiscard]] static auto get_space(std::vector<std::pair<T, T>> const &payload)
    {
        auto mspace = hdf5::Space::simple({ payload.size(), 2U });
        mspace.select_all();
        auto fspace = hdf5::Space::simple({ payload.size(), 2U });
        fspace.select_all();
        return std::make_pair(std::move(mspace), std::move(fspace));
    }
    [[nodiscard]] static auto get_space(std::vector<Scalar> const &payload) -> std::pair<hdf5::Space, hdf5::Space>;
    [[nodiscard]] static auto get_space(std::vector<MFAVector> const &payload) -> std::pair<hdf5::Space, hdf5::Space>;
    // exclude the off-diag components
    [[nodiscard]] static auto get_space(std::vector<MFATensor> const &payload) -> std::pair<hdf5::Space, hdf5::Space>;
    [[nodiscard]] static auto get_space(std::vector<CurviCoord> const &payload) -> std::pair<hdf5::Space, hdf5::Space>;
    [[nodiscard]] static auto get_space(std::vector<Particle::PSD> const &payload) -> std::pair<hdf5::Space, hdf5::Space>;
};
HYBRID1D_END_NAMESPACE
