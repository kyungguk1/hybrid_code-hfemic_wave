/*
 * Copyright (c) 2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>
#include <map>

LIBPIC_NAMESPACE_BEGIN(1)
/// Counter beam velocity distribution function
/// \details
/// f(v, α) = exp(-(x - xs)^2)*exp(-sin^2α/ν^2)/(2π θ^3 A(xs) B(ν)),
///
/// where x = v/θ and ν = √(B/B0) ν0;
/// A(b) = (1/2) * (b exp(-b^2) + √π (1/2 + b^2) erfc(-b));
/// B(ν) = 2ν F(1/ν),
/// where F(x) is Dawson's integral F.
///
class CounterBeamVDF : public VDF<CounterBeamVDF> {
    using Super = VDF<CounterBeamVDF>;

    struct Params {
        Real vth;
        Real vth_cubed;
        Real xs;        // vs normalized by vth
        Real Ab;        // normalization constant associated with velocity distribution
        Real T_by_vth2; // total temperature normalized by vth^2
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
    /// Construct a counter beam distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A CounterBeamPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    CounterBeamVDF(CounterBeamPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

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
        auto const &[T1OT, T2OT] = T1_and_T2_by_T(pos);
        MFATensor  vv{ T1OT, T2OT, T2OT, 0, 0, 0 }; // field-aligned 2nd moment
        auto const nT = Real{ this->n0(pos) } * m_physical.T_by_vth2 * (m_physical.vth * m_physical.vth);
        return geomtr.mfa_to_cart(vv *= nT, pos);
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
    [[nodiscard]] auto T1_and_T2_by_T(CurviCoord const &) const noexcept -> std::pair<Real, Real>;
    [[nodiscard]] Real eta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real nu(CurviCoord const &) const noexcept;
    [[nodiscard]] Real Fv_of_x(Real) const noexcept;
    [[nodiscard]] Real N_of_q1(Real) const;
    [[nodiscard]] Real q1_of_N(Real) const;
    [[nodiscard]] Real x_of_Fv(Real) const;

    [[nodiscard]] static auto f_common(MFAVector const &v_by_vth, Real nu, Params const &, Real denom) noexcept -> Real;

    [[nodiscard]] Particle load() const;

public:
    class [[nodiscard]] RejectionSampling {
        static constexpr Real threshold_factor = 4;

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
