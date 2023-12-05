/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Recorder.h"

#include <stdexcept>

HYBRID1D_BEGIN_NAMESPACE
Recorder::Recorder(std::pair<int, Range> const recording_frequency, parallel::mpi::Comm _subdomain_comm, parallel::mpi::Comm const &world_comm)
: m_recording_frequency{ recording_frequency.first * long{ Input::inner_Nt } }
, m_recording_temporal_extent{ recording_frequency.second }
, subdomain_comm{ std::move(_subdomain_comm) }
{
    if (!subdomain_comm->operator bool())
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid subdomain_comm" };
    if (!world_comm)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid world_comm" };
    m_is_world_master = master == world_comm.rank();
}
bool Recorder::should_record_at(const long step_count) const noexcept
{
    return ((m_recording_frequency > 0) && (0 == step_count % m_recording_frequency))
        && (m_recording_temporal_extent.len <= 0 || m_recording_temporal_extent.is_member(step_count * Input::dt));
}

auto Recorder::get_space(std::vector<Scalar> const &payload) -> std::pair<hdf5::Space, hdf5::Space>
{
    static_assert(sizeof(Scalar) % sizeof(Real) == 0);
    static_assert(sizeof(Scalar) / sizeof(Real) == 1);

    auto mspace = hdf5::Space::simple(payload.size());
    mspace.select_all();

    auto fspace = hdf5::Space::simple(payload.size());
    fspace.select_all();

    return std::make_pair(mspace, fspace);
}
auto Recorder::get_space(std::vector<MFAVector> const &payload) -> std::pair<hdf5::Space, hdf5::Space>
{
    constexpr auto size = 3U;
    static_assert(sizeof(MFAVector) % sizeof(Real) == 0);
    static_assert(sizeof(MFAVector) / sizeof(Real) == size);

    auto mspace = hdf5::Space::simple({ payload.size(), size });
    mspace.select_all();

    auto fspace = hdf5::Space::simple({ payload.size(), size });
    fspace.select_all();

    return std::make_pair(mspace, fspace);
}
auto Recorder::get_space(std::vector<MFATensor> const &payload) -> std::pair<hdf5::Space, hdf5::Space>
{
    static_assert(sizeof(MFATensor) % sizeof(Real) == 0);
    static_assert(sizeof(MFATensor) / sizeof(Real) == 6);

    auto mspace = hdf5::Space::simple({ payload.size(), sizeof(MFATensor) / sizeof(Real) });
    // diagonal
    mspace.select(H5S_SELECT_SET, { 0U, 0U }, { payload.size(), 3U });
    // off-diag
    // mspace.select(H5S_SELECT_OR, { 0U, 3U }, { payload.size(), 3U });

    auto fspace = hdf5::Space::simple({ payload.size(), 3U });
    fspace.select_all();

    return std::make_pair(mspace, fspace);
}
auto Recorder::get_space(std::vector<CurviCoord> const &payload) -> std::pair<hdf5::Space, hdf5::Space>
{
    constexpr auto size = 1U;
    static_assert(sizeof(CurviCoord) % sizeof(Real) == 0);
    static_assert(sizeof(CurviCoord) / sizeof(Real) == size);

    auto mspace = hdf5::Space::simple(payload.size());
    mspace.select_all();

    auto fspace = hdf5::Space::simple(payload.size());
    fspace.select_all();

    return std::make_pair(mspace, fspace);
}
auto Recorder::get_space(std::vector<Particle::PSD> const &payload) -> std::pair<hdf5::Space, hdf5::Space>
{
    constexpr auto size = 3U;
    static_assert(sizeof(Particle::PSD) % sizeof(Real) == 0);
    static_assert(sizeof(Particle::PSD) / sizeof(Real) == size);

    auto mspace = hdf5::Space::simple({ payload.size(), size });
    mspace.select_all();

    auto fspace = hdf5::Space::simple({ payload.size(), size });
    fspace.select_all();

    return std::make_pair(mspace, fspace);
}
HYBRID1D_END_NAMESPACE
