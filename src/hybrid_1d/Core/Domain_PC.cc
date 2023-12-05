/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Domain_PC.h"
#include "Domain.hh"

HYBRID1D_BEGIN_NAMESPACE
Domain_PC::Domain_PC(ParamSet const &params, Delegate *delegate)
: Domain{ params, delegate }
, bfield_1{ params }
, efield_1{ params }
, part_predict{ params }
, cold_predict{ params }
, source_predict{ params }
{
}

void Domain_PC::advance_by(unsigned const n_steps)
{
    Domain &domain = *this;

    // pre-process
    //
    if (!is_recurring_pass) { // execute only once
        is_recurring_pass = true;
        delegate->once(domain);
        //
        // fill in ghost cells
        //
        delegate->boundary_pass(domain, efield);
        delegate->boundary_pass(domain, bfield);
        //
        // deposit charge and current densities
        //
        charge.reset();
        current.reset();
        for (PartSpecies &sp : part_species) {
            sp.collect_part();
            charge += collect_smooth(rho, sp);
            current += collect_smooth(J, sp);
        }
        for (ColdSpecies &sp : cold_species) {
            delegate->boundary_pass(domain, sp);
            sp.collect_part();
            charge += collect_smooth(rho, sp);
            current += collect_smooth(J, sp);
        }
        for (ExternalSource const &sp : external_sources) {
            auto &predictor = source_predict = sp;
            predictor.update(0);
            charge += collect_smooth(rho, predictor);
            current += collect_smooth(J, predictor);
        }
    }

    // cycle
    //
    for (long i = 0; i < n_steps; ++i) {
        delegate->prologue(domain, i);
        cycle(domain);
        delegate->epilogue(domain, i);
    }

    // post-process; collect all moments
    //
    for (PartSpecies &sp : part_species) {
        sp.collect_all();
        delegate->boundary_gather(domain, sp);
    }
    for (ColdSpecies &sp : cold_species) {
        sp.collect_all();
        // this is to collect moments from, if any, worker threads
        delegate->boundary_gather(domain, sp);
    }
}
void Domain_PC::cycle(Domain const &domain)
{
    predictor_step(domain);
    corrector_step(domain);
}
void Domain_PC::predictor_step(Domain const &domain)
{
    BField     &bfield_0 = this->bfield;
    EField     &efield_0 = this->efield;
    Real const &dt       = params.dt;
    //
    // 1. Faraday's law; predict 1
    //
    bfield_1 = bfield_0;
    bfield_1.update(efield_0, dt);
    delegate->boundary_pass(domain, bfield_1);
    //
    // 2. Ohm's law; predict 1
    //
    efield_1.update(bfield_1, charge, current);
    delegate->boundary_pass(domain, efield_1);
    //
    // 3. Average fields
    //
    (bfield_1 += bfield_0) *= CartVector{ .5 };
    (efield_1 += efield_0) *= CartVector{ .5 };
    //
    // 4 & 5. Particle push and deposit charge and current densities; predict
    //
    charge.reset();
    current.reset();
    for (PartSpecies const &sp : part_species) {
        auto &predictor = part_predict = sp;

        predictor.update_pos(0.5 * dt, 0.5);
        delegate->boundary_pass(domain, predictor);

        predictor.update_vel(bfield_1, efield_1, dt);

        predictor.update_pos(0.5 * dt, 0.5);
        delegate->boundary_pass(domain, predictor);

        predictor.collect_part();
        charge += collect_smooth(rho, predictor);
        current += collect_smooth(J, predictor);
    }
    for (ColdSpecies const &sp : cold_species) {
        auto &predictor = cold_predict = sp;

        predictor.update_vel(bfield_1, efield_1, dt);

        predictor.collect_part();
        charge += collect_smooth(rho, predictor);
        current += collect_smooth(J, predictor);
    }
    for (ExternalSource const &sp : external_sources) {
        auto &predictor = source_predict = sp;

        predictor.update(dt); // at full-time step

        charge += collect_smooth(rho, predictor);
        current += collect_smooth(J, predictor);
    }
}
void Domain_PC::corrector_step(Domain const &domain)
{
    BField     &bfield_0 = this->bfield;
    EField     &efield_0 = this->efield;
    Real const &dt       = params.dt;
    //
    // 6. Faraday's law; predict 2
    //
    bfield_1 = bfield_0;
    bfield_1.update(efield_1, dt);
    delegate->boundary_pass(domain, bfield_1);
    //
    // 7. Ohm's law; predict 2
    //
    efield_1.update(bfield_1, charge, current);
    delegate->boundary_pass(domain, efield_1);
    //
    // 8. Average fields
    //
    (bfield_1 += bfield_0) *= CartVector{ .5 };
    (efield_1 += efield_0) *= CartVector{ .5 };
    //
    // 9 & 10. Particle push and deposit charge and current densities; correct
    //
    charge.reset();
    current.reset();
    for (PartSpecies &sp : part_species) {
        sp.update_pos(0.5 * dt, 0.5);
        delegate->boundary_pass(domain, sp);

        sp.update_vel(bfield_1, efield_1, dt);

        sp.update_pos(0.5 * dt, 0.5);
        delegate->boundary_pass(domain, sp);

        sp.collect_part();
        charge += collect_smooth(rho, sp);
        current += collect_smooth(J, sp);
    }
    for (ColdSpecies &sp : cold_species) {
        sp.update_vel(bfield_1, efield_1, dt);

        sp.collect_part();
        charge += collect_smooth(rho, sp);
        current += collect_smooth(J, sp);
    }
    for (ExternalSource &sp : external_sources) {
        sp.update(dt); // at full-time step

        charge += collect_smooth(rho, sp);
        current += collect_smooth(J, sp);
    }
    //
    // 11. Faraday's law; correct
    //
    bfield_0.update(efield_1, dt);
    delegate->boundary_pass(domain, bfield_0);
    //
    // 12. Ohm's law; correct
    //
    efield_0.update(bfield_0, charge, current);
    delegate->boundary_pass(domain, efield_0);
}
HYBRID1D_END_NAMESPACE
