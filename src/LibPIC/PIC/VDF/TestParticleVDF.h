/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>

#include <algorithm>

LIBPIC_NAMESPACE_BEGIN(1)
/// Test particle VDF
/// \details The sole job of this object is to dispense particles initialized with
///          the velocity and position passed by a TestParticleDesc object.
///
///          When all particles are exhausted, any further inquiry to the emit functions
///          returns a default-constructed Particle object which callers should ignore.
class TestParticleVDF : public VDF<TestParticleVDF> {
    using Super = VDF<TestParticleVDF>;

    KineticPlasmaDesc             m_desc;
    mutable std::vector<Particle> m_particles; // holder for remaining particles
public:
    unsigned initial_number_of_test_particles;

    /// Construct a test particle generator
    /// \tparam N The number of test particles.
    /// \param desc A TestParticleDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. (Not used here.)
    template <unsigned N>
    TestParticleVDF(TestParticleDesc<N> const &desc, Geometry const &geo, Range const &domain_extent, [[maybe_unused]] Real c)
    : VDF{ geo, domain_extent }, m_desc{ desc }, m_particles(N), initial_number_of_test_particles{ N }
    {
        std::transform(
            begin(desc.vel), end(desc.vel), begin(desc.pos), rbegin(m_particles) /*reverse order*/,
            [&geo](MFAVector const &vel, CurviCoord const &pos) -> Particle {
                return { geo.mfa_to_cart(vel, pos), pos };
            });
    }

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->m_desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &) const { return Scalar{ 0 }; }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &) const { return CartVector{ 0, 0, 0 }; }
    [[nodiscard]] inline auto impl_nvv(Badge<Super>, CurviCoord const &) const { return CartTensor{ 0, 0, 0, 0, 0, 0 }; }

    [[nodiscard]] inline Real impl_Nrefcell_div_Ntotal(Badge<Super>) const { return 1; }
    [[nodiscard]] inline Real impl_f(Badge<Super>, Particle const &ptl) const { return f0(ptl); }

    [[nodiscard]] auto impl_emit(Badge<Super>, unsigned long) const -> std::vector<Particle>;
    [[nodiscard]] auto impl_emit(Badge<Super>) const -> Particle;

    // equilibrium physical distribution function
    //
    [[nodiscard]] Real f0(CartVector const &, CurviCoord const &) const noexcept { return 0; }
    [[nodiscard]] Real f0(Particle const &ptl) const noexcept { return f0(ptl.vel, ptl.pos); }

    // marker particle distribution function
    //
    [[nodiscard]] Real g0(CartVector const &, CurviCoord const &) const noexcept { return 1; }
    [[nodiscard]] Real g0(Particle const &ptl) const noexcept { return g0(ptl.vel, ptl.pos); }

private:
    [[nodiscard]] Particle load() const;
};
LIBPIC_NAMESPACE_END(1)
