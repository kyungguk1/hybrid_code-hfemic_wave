/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/RelativisticVDF.h>
#include <map>

LIBPIC_NAMESPACE_BEGIN(1)
/// Relativistic loss-cone velocity distribution function
/// \details
/// f0(u1, u2) = n0*exp(-x1^2)/(π^3/2 vth1 vth2^2) * (exp(-x2^2) - exp(-x2^2/β))/(1 - β),
///
/// where u = γv, x1 = u1/vth1, x2 = v2/vth2.
/// The effective temperature in the perpendicular direction is 2*T2/vth2^2 = 1 + β
///
class RelativisticLossconeVDF : public RelativisticVDF<RelativisticLossconeVDF> {
    using Super = RelativisticVDF<RelativisticLossconeVDF>;

    struct RejectionSampler { // rejection sampler
        RejectionSampler() noexcept = default;
        explicit RejectionSampler(Real beta /*must not be 1*/);
        [[nodiscard]] Real sample() const noexcept;
        [[nodiscard]] Real fOg(Real x) const noexcept; // ratio of the target to the proposed distributions
        //
        static constexpr Real Delta{ 0 };     //!< Δ parameter.
        Real                  beta;           //!< β parameter.
        Real                  alpha;          //!< thermal spread of of the proposed distribution
        Real                  M;              //!< the ratio f(x_pk)/g(x_pk)
        static constexpr Real a_offset = 0.3; //!< optimal value for thermal spread of the proposed distribution
    };
    struct Params {
        Real losscone_beta; // loss-cone beta.
        Real vth1;          //!< Parallel thermal speed.
        Real vth1_cubed;    //!< vth1^3.
        Real xth2_square;   //!< The ratio of vth2^2 to vth1^2.
        Params() noexcept = default;
        Params(Real losscone_beta, Real vth1, Real T2OT1) noexcept;
    };
    static constexpr Real eps = 1e-10;

    LossconePlasmaDesc desc;
    //
    Params m_physical_eq;
    Params m_marker_eq;
    // marker psd parallel thermal speed
    Range m_N_extent;
    Real  m_Nrefcell_div_Ntotal;
    //
    std::map<Real, Real> m_q1_of_N;

public:
    /// Construct a relativistic loss-cone distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A BiMaxPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    RelativisticLossconeVDF(LossconePlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &pos) const -> Scalar
    {
        return particle_flux_vector(pos).t / c;
    }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &pos) const -> CartVector
    {
        return geomtr.mfa_to_cart(particle_flux_vector(pos).s, pos);
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
    [[nodiscard]] Real vth1(CurviCoord const &) const noexcept { return m_physical_eq.vth1; }
    [[nodiscard]] Real vth1_cubed(CurviCoord const &) const noexcept { return m_physical_eq.vth1_cubed; }
    [[nodiscard]] Real xth2_square(CurviCoord const &pos) const noexcept { return m_physical_eq.xth2_square * eta(pos); }
    [[nodiscard]] Real marker_vth1(CurviCoord const &) const noexcept { return m_marker_eq.vth1; }
    [[nodiscard]] Real marker_vth1_cubed(CurviCoord const &) const noexcept { return m_marker_eq.vth1_cubed; }
    [[nodiscard]] Real losscone_beta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real eta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real eta_b(CurviCoord const &) const noexcept;
    [[nodiscard]] Real N_of_q1(Real q1) const noexcept;
    [[nodiscard]] Real q1_of_N(Real N) const;

    // particle flux four-vector in co-moving frame
    [[nodiscard]] auto particle_flux_vector(CurviCoord const &) const -> FourMFAVector;
    // stress-energy four-tensor in co-moving frame
    [[nodiscard]] auto stress_energy_tensor(CurviCoord const &) const -> FourMFATensor;

    [[nodiscard]] auto load() const -> Particle;

    // velocity is normalized by vth1 and shifted to drifting plasma frame
    [[nodiscard]] static auto f_common(MFAVector const &g_vel, Real xth2_square, Real losscone_beta, Real denom) noexcept -> Real;
};
LIBPIC_NAMESPACE_END(1)
