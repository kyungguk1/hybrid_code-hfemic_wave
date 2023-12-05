/*
 * Copyright (c) 2022-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Species.h"

#include <limits>
#include <vector>

HYBRID1D_BEGIN_NAMESPACE
/// external current source
///
class ExternalSource : public Species {
    ExternalSourceBase      src_desc;
    std::vector<CurviCoord> src_pos;
    std::vector<MFAVector>  src_Jre; // in field-aligned coordinates
    std::vector<MFAVector>  src_Jim; // in field-aligned coordinates
    unsigned                number_of_source_points;
    long                    m_cur_step{ std::numeric_limits<long>::quiet_NaN() };
    struct RampSlope {
        Real ease_in{ std::numeric_limits<Real>::quiet_NaN() };
        Real ease_out{ std::numeric_limits<Real>::quiet_NaN() };
    } ramp_slope{};

public:
    [[nodiscard]] PlasmaDesc const *operator->() const noexcept override { return &src_desc; }

    [[nodiscard]] Real charge_density_conversion_factor() const noexcept override { return 0; }
    [[nodiscard]] Real current_density_conversion_factor() const noexcept override { return 1; }
    [[nodiscard]] Real energy_density_conversion_factor() const noexcept override { return 0; }

    ExternalSource &operator=(ExternalSource const &) = default;
    template <unsigned N>
    ExternalSource(ParamSet const &params, ExternalSourceDesc<N> const &src);
    ExternalSource() = default; // needed for empty std::array
    explicit ExternalSource(ParamSet const &params)
    : Species{ params } {} // needed for Domain_PC

    /// Reset the current step count
    /// \param step_count Simulation step count.
    void               set_cur_step(long step_count) noexcept { this->m_cur_step = step_count; }
    [[nodiscard]] long cur_step() const noexcept { return m_cur_step; }

    /// Update the moments at a given time
    /// \note After this call, the current step count is incremented.
    ///       Therefore, this should be called only once at each cycle.
    /// \param delta_t Time (delta from cur_step * params.dt) at which the moments should be calculated.
    void update(Real delta_t);

#ifndef DEBUG
private:
#endif
    void               collect(Grid<Scalar> &rho, Grid<CartVector> &J, Real t) const;
    [[nodiscard]] auto current(MFAVector const &J0re, MFAVector const &J0im, Real t) const noexcept -> MFAVector;
    [[nodiscard]] auto envelope(Real t) const noexcept -> Real;

    // attribute export facility
    //
    template <class Object>
    friend auto write_attr(Object &obj, ExternalSource const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Group &obj, ExternalSource const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Dataset &obj, ExternalSource const &sp) -> decltype(obj);
    friend auto operator<<(hdf5::Group &&obj, ExternalSource const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
    friend auto operator<<(hdf5::Dataset &&obj, ExternalSource const &sp) -> decltype(obj)
    {
        return std::move(obj << sp);
    }
};
HYBRID1D_END_NAMESPACE
