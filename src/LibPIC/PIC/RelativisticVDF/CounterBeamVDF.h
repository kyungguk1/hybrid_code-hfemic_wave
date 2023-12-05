/*
 * Copyright (c) 2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/RelativisticVDF.h>
#include <map>

LIBPIC_NAMESPACE_BEGIN(1)
/// Counter beam velocity distribution function
/// \details
/// f0(u, α) = exp(-(x - xs)^2)*exp(-sin^2α/ν^2)/(2π θ^3 A(xs) B(ν)),
///
/// where u = γv, x = u/θ and ν = √(B/B0) ν0;
/// A(b) = (1/2) * (b exp(-b^2) + √π (1/2 + b^2) erfc(-b));
/// B(ν) = 2ν F(1/ν),
/// where F(x) is Dawson's integral F.
///
class RelativisticCounterBeamVDF : public RelativisticVDF<RelativisticCounterBeamVDF> {
    using Super = RelativisticVDF<RelativisticCounterBeamVDF>;

    struct Params {
        Real vth;
        Real vth_cubed;
        Real xs; // vs normalized by vth
        Real Ab; // normalization constant associated with velocity distribution
        //
        [[nodiscard]] static Real Bnu(Real nu) noexcept;

        Params() noexcept = default;
        Params(Real vth, Real vs) noexcept;
    };

    CounterBeamPlasmaDesc desc;
    //
    Real   nu0; // equatorial value for pitch angle gaussian distribution
    Params m_physical;
    Params m_marker;
    //
    Range m_N_extent;
    Range m_Fv_extent;
    Real  m_Nrefcell_div_Ntotal;
    //
    std::map<Real, Real> m_q1_of_N;
    std::map<Real, Real> m_x_of_Fv;

public:
    /// Construct a relativistic counter beam distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A CounterBeamPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    RelativisticCounterBeamVDF(CounterBeamPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &pos) const -> Scalar
    {
        return particle_flux_vector(pos).t / c;
    }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &) const -> CartVector
    {
        return { 0, 0, 0 };
    }
    [[nodiscard]] inline auto impl_nuv(Badge<Super>, CurviCoord const &pos) const -> FourCartTensor
    {
        return geomtr.mfa_to_cart(stress_energy_tensor(pos), pos);
    }

    [[nodiscard]] inline Real impl_Nrefcell_div_Ntotal(Badge<Super>) const { return m_Nrefcell_div_Ntotal; }
    [[nodiscard]] inline Real impl_f(Badge<Super>, Particle const &ptl) const { return f0(ptl); }

    [[nodiscard]] auto impl_emit(Badge<Super>, unsigned long) const -> std::vector<Particle>;
    [[nodiscard]] auto impl_emit(Badge<Super>) const -> Particle;

    // equilibrium physical distribution function
    //
    [[nodiscard]] Real f0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real f0(Particle const &ptl) const noexcept { return f0(ptl.gcgvel, ptl.pos); }

    // marker particle distribution function
    //
    [[nodiscard]] Real g0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real g0(Particle const &ptl) const noexcept { return g0(ptl.gcgvel, ptl.pos); }

private:
    [[nodiscard]] Real nu(CurviCoord const &) const noexcept;
    [[nodiscard]] Real eta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real Fv_of_x(Real) const noexcept;
    [[nodiscard]] Real N_of_q1(Real) const;
    [[nodiscard]] Real q1_of_N(Real) const;
    [[nodiscard]] Real x_of_Fv(Real) const;

    // particle flux four-vector in co-moving frame
    [[nodiscard]] auto particle_flux_vector(CurviCoord const &) const -> FourMFAVector;
    // stress-energy four-tensor in co-moving frame
    [[nodiscard]] auto stress_energy_tensor(CurviCoord const &) const -> FourMFATensor;

    [[nodiscard]] auto load() const -> Particle;

    // velocity is normalized by vth1 and shifted to drifting plasma frame
    [[nodiscard]] static auto f_common(MFAVector const &g_vel, Real nu, Params const &, Real denom) noexcept -> Real;

public:
    static constexpr Real threshold_factor = 4;
    class [[nodiscard]] RejectionSampling {
        Real nu2;
        Real a_max;
        Real f_peak;

        [[nodiscard]] inline Real fa(Real) const noexcept;
        [[nodiscard]] inline Real draw() const &noexcept;

    public:
        explicit RejectionSampling(Real nu) noexcept;
        [[nodiscard]] Real sample() &&noexcept;
    };
};
LIBPIC_NAMESPACE_END(1)
