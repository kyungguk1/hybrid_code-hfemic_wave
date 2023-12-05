/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>
#include <map>

LIBPIC_NAMESPACE_BEGIN(1)
/// Loss-cone velocity distribution function
/// \details
/// f(v1, v2) = exp(-x1^2)/(π^3/2 vth1 vth2^2)/(1 - β) * (exp(-x2^2) - exp(-x2^2/β)),
///
/// where x1 = v1/vth1 and x2 = v2/vth2.
/// The effective temperature in the perpendicular direction is 2*T2/vth2^2 = 1 + β
///
class LossconeVDF : public VDF<LossconeVDF> {
    using Super = VDF<LossconeVDF>;

    struct RejectionSampler { // rejection sampler
        explicit RejectionSampler(Real beta /*must not be 1*/);
        [[nodiscard]] Real sample() const noexcept;
        // ratio of the target to the proposed distributions
        [[nodiscard]] Real fOg(Real x) const noexcept;
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
    //
    Range m_N_extent;
    Real  m_Nrefcell_div_Ntotal;
    //
    std::map<Real, Real> m_q1_of_N;

public:
    /// Construct a loss-cone distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A BiMaxPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    LossconeVDF(LossconePlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &pos) const -> Scalar
    {
        constexpr Real n0_eq   = 1;
        auto const     beta_eq = m_physical_eq.losscone_beta;
        return n0_eq * (eta(pos) - beta_eq * eta_b(pos)) / (1 - beta_eq);
    }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &) const -> CartVector
    {
        return { 0, 0, 0 };
    }
    [[nodiscard]] inline auto impl_nvv(Badge<Super>, CurviCoord const &pos) const -> CartTensor
    {
        Real const T2OT1 = (1 + losscone_beta(pos)) * xth2_square(pos);
        MFATensor  vv{ 1, T2OT1, T2OT1, 0, 0, 0 }; // field-aligned 2nd moment
        return geomtr.mfa_to_cart(vv *= .5 * vth1(pos) * vth1(pos), pos) * Real{ this->n0(pos) };
    }

    [[nodiscard]] inline Real impl_Nrefcell_div_Ntotal(Badge<Super>) const { return m_Nrefcell_div_Ntotal; }
    [[nodiscard]] inline Real impl_f(Badge<Super>, Particle const &ptl) const { return f0(ptl); }

    [[nodiscard]] auto impl_emit(Badge<Super>, unsigned long) const -> std::vector<Particle>;
    [[nodiscard]] auto impl_emit(Badge<Super>) const -> Particle;

    // equilibrium physical distribution function
    //
    [[nodiscard]] Real f0(CartVector const &vel, CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real f0(Particle const &ptl) const noexcept { return f0(ptl.vel, ptl.pos); }

    // marker particle distribution function
    //
    [[nodiscard]] Real g0(CartVector const &vel, CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real g0(Particle const &ptl) const noexcept { return g0(ptl.vel, ptl.pos); }

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

    [[nodiscard]] Particle load() const;

    // velocity is normalized by vth1
    [[nodiscard]] static auto f_common(MFAVector const &vel, Real xth2_square, Real losscone_beta, Real denom) noexcept;
};
LIBPIC_NAMESPACE_END(1)
