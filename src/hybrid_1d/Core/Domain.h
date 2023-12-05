/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../ParamSet.h"
#include "BField.h"
#include "Charge.h"
#include "ColdSpecies.h"
#include "Current.h"
#include "EField.h"
#include "ExternalSource.h"
#include "PartSpecies.h"
#include "Species.h"

#include <array>
#include <tuple>
#include <utility>

HYBRID1D_BEGIN_NAMESPACE
class Delegate;

class Domain {
public:
    ParamSet const                                          params;
    Delegate *const                                         delegate;
    BField                                                  bfield;
    EField                                                  efield;
    Charge                                                  charge;
    Current                                                 current;
    std::array<PartSpecies, ParamSet::part_indices::size()> part_species;
    std::array<ColdSpecies, ParamSet::cold_indices::size()> cold_species;
    //
    std::array<ExternalSource, ParamSet::source_indices::size()> external_sources;

protected:
    Charge  rho;
    Current J;
    bool    is_recurring_pass{ false };

public:
    virtual ~Domain();
    virtual void advance_by(unsigned n_steps) = 0;

protected:
    Domain(ParamSet const &params, Delegate *delegate);

    template <class Species>
    Charge const &collect_smooth(Charge &rho, Species const &sp) const;
    template <class Species>
    Current const &collect_smooth(Current &J, Species const &sp) const;

private:
    template <class... Ts, class Int, Int... Is>
    static auto make_part_species(ParamSet const &params, std::tuple<Ts...> const &descs, std::integer_sequence<Int, Is...>);
    template <class... Ts, class Int, Int... Is>
    static auto make_cold_species(ParamSet const &params, std::tuple<Ts...> const &descs, std::integer_sequence<Int, Is...>);
    template <class... Ts, class Int, Int... Is>
    static auto make_external_sources(ParamSet const &params, std::tuple<Ts...> const &descs, std::integer_sequence<Int, Is...>);
};
HYBRID1D_END_NAMESPACE
