/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Recorder.h"

#include <fstream>
#include <string_view>

HYBRID1D_BEGIN_NAMESPACE
/// spatial average of field and ion energy density recorder
/// field-aligned components are recorded;
/// suffix 1, 2, and 3 means three field-aligned components:
///     1 : parallel, 2 : perpendicular, and 3 : out-of-plane
///
class EnergyRecorder : public Recorder {
    [[nodiscard]] auto filepath(std::string_view const &wd) const;

    std::ofstream os;

public:
    EnergyRecorder(ParamSet const &params, parallel::mpi::Comm subdomain_comm, parallel::mpi::Comm const &world_comm);

private:
    void record(Domain const &domain, long step_count) override;

    [[nodiscard]] static auto dump(BField const &bfield) -> MFAVector;
    [[nodiscard]] static auto dump(EField const &efield) -> MFAVector;
    [[nodiscard]] static auto dump(Species const &sp) -> std::vector<MFAVector>;
};
HYBRID1D_END_NAMESPACE
