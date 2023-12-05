/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"

HYBRID1D_BEGIN_NAMESPACE
class Species;

/// charge density
///
class Charge : public Grid<Scalar> {
    Grid<Scalar> buffer;

public:
    ParamSet const params;

    virtual ~Charge() = default;
    explicit Charge(ParamSet const &);
    Charge &operator=(ParamSet const &) = delete;

    [[nodiscard]] auto &grid_whole_domain_extent() const noexcept { return params.half_grid_whole_domain_extent; }
    [[nodiscard]] auto &grid_subdomain_extent() const noexcept { return params.half_grid_subdomain_extent; }

    void reset() &noexcept { this->fill_all(Scalar{}); }
    void smooth() &noexcept { this->swap(buffer.smooth_assign(*this)); }

    virtual Charge &operator+=(Species const &) noexcept;
};

/// Λ
///
class Lambda : public Charge {
    using Charge::smooth;

public:
    using Charge::Charge;
    Lambda &operator+=(Species const &) noexcept override;
};
HYBRID1D_END_NAMESPACE
