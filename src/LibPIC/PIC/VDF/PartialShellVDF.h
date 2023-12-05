/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>
#include <map>

LIBPIC_NAMESPACE_BEGIN(1)
/// Partial shell velocity distribution function
/// \details
/// f(v, α) = exp(-(x - xs)^2)*sin^ζ(α)/(2π θ^3 A(xs) B(ζ)),
///
/// where x = v/θ;
/// A(b) = (1/2) * (b exp(-b^2) + √π (1/2 + b^2) erfc(-b));
/// B(ζ) = √π Γ(1 + ζ/2)/Γ(1.5 + ζ/2).
///
class PartialShellVDF : public VDF<PartialShellVDF> {
    using Super = VDF<PartialShellVDF>;

    struct Params {
        long zeta;
        Real vth;
        Real vth_cubed;
        Real xs;         // vs normalized by vth
        Real Ab;         // normalization constant associated with velocity distribution
        Real Bz;         // normalization constant associated with pitch angle distribution
        Real T1_by_vth2; // parallel temperature normalized by thermal speed squared

        Params() noexcept = default;
        Params(Real vth, unsigned zeta, Real vs) noexcept;
    };

    PartialShellPlasmaDesc desc;
    //
    Params m_physical; // no eq subscript because shell parameters are invariant with latitude
    Params m_marker;   // no eq subscript because shell parameters are invariant with latitude
    //
    Range m_N_extent;
    Real  m_Nrefcell_div_Ntotal;
    Range m_Fv_extent;
    Range m_Fa_extent;
    //
    std::map<Real, Real> m_q1_of_N;
    std::map<Real, Real> m_x_of_Fv;
    std::map<Real, Real> m_a_of_Fa;

public:
    /// Construct a partial shell distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A PartialShellPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    PartialShellVDF(PartialShellPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

    // VDF interfaces
    //
    [[nodiscard]] inline decltype(auto) impl_plasma_desc(Badge<Super>) const noexcept { return (this->desc); }

    [[nodiscard]] inline auto impl_n(Badge<Super>, CurviCoord const &pos) const -> Scalar
    {
        constexpr Real n0_eq = 1;
        return n0_eq * eta(pos);
    }
    [[nodiscard]] inline auto impl_nV(Badge<Super>, CurviCoord const &) const -> CartVector
    {
        return { 0, 0, 0 };
    }
    [[nodiscard]] inline auto impl_nvv(Badge<Super>, CurviCoord const &pos) const -> CartTensor
    {
        Real const T2OT1 = Real(2 + m_physical.zeta) / 2;
        MFATensor  vv{ 1, T2OT1, T2OT1, 0, 0, 0 }; // field-aligned 2nd moment
        auto const T1 = m_physical.T1_by_vth2 * (m_physical.vth * m_physical.vth);
        return geomtr.mfa_to_cart(vv *= T1, pos) * Real{ this->n0(pos) };
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
    [[nodiscard]] Real eta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real N_of_q1(Real) const noexcept;
    [[nodiscard]] Real Fa_of_a(Real) const noexcept;
    [[nodiscard]] Real Fv_of_x(Real) const noexcept;
    [[nodiscard]] Real q1_of_N(Real) const;
    [[nodiscard]] Real x_of_Fv(Real) const;
    [[nodiscard]] Real a_of_Fa(Real) const;

    [[nodiscard]] Particle load() const;

    // velocity is normalized by vth
    [[nodiscard]] static auto f_common(MFAVector const &vel, Params const &, Real denom) noexcept -> Real;
};
LIBPIC_NAMESPACE_END(1)
