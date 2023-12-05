/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"

HYBRID1D_BEGIN_NAMESPACE
class BField;
class EField;
class Lambda;
class Species;
class Gamma;

/// current density
///
class Current : public Grid<CartVector> {
    Grid<CartVector> buffer;

public:
    ParamSet const params;
    Geometry const geomtr;

    virtual ~Current() = default;
    explicit Current(ParamSet const &);
    Current &operator=(ParamSet const &) = delete;

    [[nodiscard]] auto &grid_whole_domain_extent() const noexcept { return params.half_grid_whole_domain_extent; }
    [[nodiscard]] auto &grid_subdomain_extent() const noexcept { return params.half_grid_subdomain_extent; }

    void reset() &noexcept { this->fill_all(CartVector{}); }
    void smooth() &noexcept { this->swap(buffer.smooth_assign(*this)); }

    virtual Current &operator+=(Species const &) noexcept;

    void advance(Lambda const &lambda, Gamma const &gamma, BField const &bfield, EField const &efield, Real dt) noexcept;

private:
    void impl_advance(Current &J, Lambda const &L, Gamma const &G, BField const &dB, EField const &E, Real dt) const noexcept;
};

/// Γ
///
class Gamma : public Current {
    using Current::advance;
    using Current::smooth;

public:
    using Current::Current;
    Gamma &operator+=(Species const &) noexcept override;
};
HYBRID1D_END_NAMESPACE
