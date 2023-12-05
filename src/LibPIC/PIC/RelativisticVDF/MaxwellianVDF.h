/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/RelativisticVDF.h>

LIBPIC_NAMESPACE_BEGIN(1)
/// Relativistic bi-Maxwellian velocity distribution function
/// \details
/// f(u1, u2) = n*exp(-x1^2 -x2^2)/(π^3/2 vth1^3 T2/T1),
///
/// where u = γv, x1 = u1/vth1, x2 = u2/(vth1*√(T2/T1))), and
/// T2 and T1 are temperatures in the directions perpendicular and
/// parallel to the background magnetic field direction, respectively.
///
class RelativisticMaxwellianVDF : public RelativisticVDF<RelativisticMaxwellianVDF> {
    using Super = RelativisticVDF<RelativisticMaxwellianVDF>;

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
    /// Construct a relativistic bi-Maxwellian distribution
    /// \note Necessary parameter check is assumed to be done already.
    /// \param desc A BiMaxPlasmaDesc object.
    /// \param geo A geometry object.
    /// \param domain_extent Spatial domain extent.
    /// \param c Light speed. A positive real.
    ///
    RelativisticMaxwellianVDF(BiMaxPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c);

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
    [[nodiscard]] Real vth1_square(CurviCoord const &) const noexcept { return m_physical_eq.vth1_square; }
    [[nodiscard]] Real marker_vth1(CurviCoord const &) const noexcept { return m_marker_eq.vth1; }
    [[nodiscard]] Real marker_vth1_cubed(CurviCoord const &) const noexcept { return m_marker_eq.vth1_cubed; }
    [[nodiscard]] Real eta(CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real T2OT1(CurviCoord const &pos) const noexcept;
    [[nodiscard]] Real N_of_q1(Real q1) const noexcept;
    [[nodiscard]] Real q1_of_N(Real N) const noexcept;

    // particle flux four-vector in co-moving frame
    [[nodiscard]] auto particle_flux_vector(CurviCoord const &) const -> FourMFAVector;
    // stress-energy four-tensor in co-moving frame
    [[nodiscard]] auto stress_energy_tensor(CurviCoord const &) const -> FourMFATensor;

    [[nodiscard]] auto load() const -> Particle;

    // velocity is normalized by vth1 and shifted to drifting plasma frame
    [[nodiscard]] static auto f_common(MFAVector const &g_vel, Real T2OT1, Real denom) noexcept -> Real;
};
LIBPIC_NAMESPACE_END(1)
