/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/RelativisticVDF.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
/// Test particle VDF
/// \details The sole job of this object is to dispense particles initialized with
///          the velocity and position passed by a TestParticleDesc object.
///
///          When all particles are exhausted, any further inquiry to the emit functions
///          returns a default-constructed Particle object which callers should ignore.
class RelativisticTestParticleVDF : public RelativisticVDF<RelativisticTestParticleVDF> {
    using Super = RelativisticVDF<RelativisticTestParticleVDF>;

    KineticPlasmaDesc             m_desc;
    mutable std::vector<Particle> m_particles; // holder for remaining particles
public:
    unsigned initial_number_of_test_particles;

    /// Construct a test particle generator
    /// \tparam N The number of test particles.
    /// \param desc A TestParticleDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    template <unsigned N>
    RelativisticTestParticleVDF(TestParticleDesc<N> const &desc, Geometry const &geo, Range const &domain_extent, Real c)
    : RelativisticVDF{ geo, domain_extent, c }, m_desc{ desc }, m_particles(N), initial_number_of_test_particles{ N }
    {
        std::transform(
            begin(desc.vel), end(desc.vel), begin(desc.pos), rbegin(m_particles) /*reverse order*/,
            [c, &geo](MFAVector const &vel, CurviCoord const &pos) -> Particle {
                if (auto const beta2 = dot(vel, vel) / (c * c); beta2 < 1) {
                    auto const beta  = std::sqrt(beta2);
                    auto const gamma = 1 / std::sqrt((1 - beta) * (1 + beta));
                    return { { gamma * c, gamma * geo.mfa_to_cart(vel, pos) }, pos };
                } else {
                    throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - test particle speed greater than or equal to c" };
                }
            });
    }

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->m_desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &) const { return Scalar{ 0 }; }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &) const { return CartVector{ 0, 0, 0 }; }
    [[nodiscard]] inline auto impl_nuv(Badge<Super>, CurviCoord const &) const { return FourCartTensor{ 0, { 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } }; }

    [[nodiscard]] inline Real impl_Nrefcell_div_Ntotal(Badge<Super>) const { return 1; }
    [[nodiscard]] inline Real impl_f(Badge<Super>, Particle const &ptl) const { return f0(ptl); }

    [[nodiscard]] auto impl_emit(Badge<Super>, unsigned long) const -> std::vector<Particle>;
    [[nodiscard]] auto impl_emit(Badge<Super>) const -> Particle;

    // equilibrium physical distribution function
    //
    [[nodiscard]] Real f0(FourCartVector const &, CurviCoord const &) const noexcept { return 0; }
    [[nodiscard]] Real f0(Particle const &ptl) const noexcept { return f0(ptl.gcgvel, ptl.pos); }

    // marker particle distribution function
    //
    [[nodiscard]] Real g0(FourCartVector const &, CurviCoord const &) const noexcept { return 1; }
    [[nodiscard]] Real g0(Particle const &ptl) const noexcept { return g0(ptl.gcgvel, ptl.pos); }

private:
    [[nodiscard]] Particle load() const;
};
LIBPIC_NAMESPACE_END(1)
