/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>

LIBPIC_NAMESPACE_BEGIN(1)
/// Bi-Maxwellian velocity distribution function
/// \details
/// f(v1, v2) = exp(-x1^2 -x2^2)/(π^3/2 vth1^3 T2/T1),
///
/// where x1 = v1/vth1, x2 = v2/(vth1*√(T2/T1))), and
/// T2 and T1 are temperatures in directions perpendicular and
/// parallel to the background magnetic field direction, respectively.
///
class MaxwellianVDF : public VDF<MaxwellianVDF> {
    using Super = VDF<MaxwellianVDF>;

    struct Params {
        Real vth1;        //!< Parallel thermal speed.
        Real T2OT1;       //!< Temperature anisotropy, T2/T1.
        Real sqrt_T2OT1;  //!< √(T2/T1).
        Real vth1_square; //!< vth1^2
        Real vth1_cubed;  //!< vth1^3

        Params() noexcept = default;
        Params(Real vth1, Real T2OT1) noexcept;
    };

    BiMaxPlasmaDesc desc;
    //
    Params m_physical_eq;
    Params m_marker_eq;
    //
    Range m_N_extent;
    Real  m_Nrefcell_div_Ntotal;

public:
    /// Construct a bi-Maxwellian distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A BiMaxPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    MaxwellianVDF(BiMaxPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

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
        Real const T2OT1 = this->T2OT1(pos);
        MFATensor  vv{ 1, T2OT1, T2OT1, 0, 0, 0 }; // field-aligned 2nd moment
        return geomtr.mfa_to_cart(vv *= .5 * vth1_square(pos), pos) * Real{ this->n0(pos) };
    }

    [[nodiscard]] inline Real impl_Nrefcell_div_Ntotal(Badge<Super>) const { return m_Nrefcell_div_Ntotal; }
    [[nodiscard]] inline Real impl_f(Badge<Super>, Particle const &ptl) const { return f0(ptl); }

    [[nodiscard]] auto impl_emit(Badge<Super>, unsigned long) const -> std::vector<Particle>;
    [[nodiscard]] auto impl_emit(Badge<Super>) const -> Particle;

    // equilibrium physical distribution function
    // f0(x1, x2, x3) = exp(-x1^2)/√π * exp(-(x2^2 + x3^2)/(T2/T1))/(π T2/T1)
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
    [[nodiscard]] Real vth1_square(CurviCoord const &) const noexcept { return m_physical_eq.vth1_square; }
    [[nodiscard]] Real marker_vth1(CurviCoord const &) const noexcept { return m_marker_eq.vth1; }
    [[nodiscard]] Real marker_vth1_cubed(CurviCoord const &) const noexcept { return m_marker_eq.vth1_cubed; }
    [[nodiscard]] Real eta(CurviCoord const &) const noexcept;
    [[nodiscard]] Real T2OT1(CurviCoord const &) const noexcept;
    [[nodiscard]] Real N_of_q1(Real q1) const noexcept;
    [[nodiscard]] Real q1_of_N(Real N) const noexcept;

    [[nodiscard]] Particle load() const;

    // velocity is normalized by vth1 and shifted to drifting plasma frame
    [[nodiscard]] static auto f_common(MFAVector const &vel, Real T2OT1, Real denom) noexcept -> Real;
};
LIBPIC_NAMESPACE_END(1)
