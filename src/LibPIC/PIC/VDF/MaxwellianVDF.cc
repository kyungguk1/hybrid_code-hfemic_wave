/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "MaxwellianVDF.h"
#include "../RandomReal.h"
#include <algorithm>
#include <cmath>

LIBPIC_NAMESPACE_BEGIN(1)
MaxwellianVDF::Params::Params(Real const vth1, Real const T2OT1) noexcept
: vth1{ vth1 }
, T2OT1{ T2OT1 }
, sqrt_T2OT1{ std::sqrt(T2OT1) }
, vth1_square{ vth1 * vth1 }
, vth1_cubed{ vth1 * vth1 * vth1 }
{
}
MaxwellianVDF::MaxwellianVDF(BiMaxPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c)
: VDF{ geo, domain_extent }, desc{ desc }
{ // parameter check is assumed to be done already
    auto const vth1 = std::sqrt(desc.beta1) * c * std::abs(desc.Oc) / desc.op;
    m_physical_eq   = { vth1, desc.T2_T1 };
    m_marker_eq     = { vth1 * std::sqrt(desc.marker_temp_ratio), desc.T2_T1 };
    //
    m_N_extent.loc        = N_of_q1(domain_extent.min());
    m_N_extent.len        = N_of_q1(domain_extent.max()) - m_N_extent.loc;
    m_Nrefcell_div_Ntotal = (N_of_q1(+0.5) - N_of_q1(-0.5)) / m_N_extent.len;
}

auto MaxwellianVDF::eta(CurviCoord const &pos) const noexcept -> Real
{
    auto const T2OT1_eq = m_physical_eq.T2OT1;
    auto const cos      = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    return 1 / (T2OT1_eq + (1 - T2OT1_eq) * cos * cos);
}
auto MaxwellianVDF::T2OT1(CurviCoord const &pos) const noexcept -> Real
{
    auto const T2OT1_eq = m_physical_eq.T2OT1;
    return T2OT1_eq * eta(pos);
}
auto MaxwellianVDF::N_of_q1(Real const q1) const noexcept -> Real
{
    if (geomtr.is_homogeneous()) {
        auto const T2OT1_eq = m_physical_eq.T2OT1;
        auto const xiD1q1   = geomtr.xi() * geomtr.D1() * q1;
        return q1 * (1 - (xiD1q1 * xiD1q1) * (T2OT1_eq - 1) / 3);
    } else {
        auto const sqrt_T2OT1_eq = m_physical_eq.sqrt_T2OT1;
        return std::atan(sqrt_T2OT1_eq * std::tan(geomtr.xi() * geomtr.D1() * q1)) / (sqrt_T2OT1_eq * geomtr.D1() * geomtr.xi());
    }
}
auto MaxwellianVDF::q1_of_N(Real const N) const noexcept -> Real
{
    if (geomtr.is_homogeneous()) {
        auto const T2OT1_eq = m_physical_eq.T2OT1;
        auto const xiD1N    = geomtr.xi() * geomtr.D1() * N;
        return N * (1 + (xiD1N * xiD1N) * (T2OT1_eq - 1) / 3);
    } else {
        auto const sqrt_T2OT1_eq = m_physical_eq.sqrt_T2OT1;
        return std::atan(std::tan(sqrt_T2OT1_eq * geomtr.D1() * geomtr.xi() * N) / sqrt_T2OT1_eq) / (geomtr.xi() * geomtr.D1());
    }
}

auto MaxwellianVDF::f_common(MFAVector const &v, Real const T2OT1, Real denom) noexcept -> Real
{
    // note that vel = {v1, v2, v3}/vth1
    // f0(x1, x2, x3) = exp(-x1^2)/√π * exp(-(x2^2 + x3^2)/(T2/T1))/(π T2/T1)
    //
    Real const f1 = std::exp(-v.x * v.x) * M_2_SQRTPI * .5;
    Real const x2 = v.y * v.y + v.z * v.z;
    Real const f2 = std::exp(-x2 / T2OT1) / (M_PI * T2OT1);
    return (f1 * f2) / denom;
}
auto MaxwellianVDF::f0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    auto const v_mfa = geomtr.cart_to_mfa(vel, pos);
    return Real{ this->n0(pos) } * f_common(v_mfa / vth1(pos), T2OT1(pos), vth1_cubed(pos));
}
auto MaxwellianVDF::g0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    auto const v_mfa = geomtr.cart_to_mfa(vel, pos);
    return Real{ this->n0(pos) } * f_common(v_mfa / marker_vth1(pos), T2OT1(pos), marker_vth1_cubed(pos));
}

auto MaxwellianVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto MaxwellianVDF::impl_emit(Badge<Super>) const -> Particle
{
    Particle ptl = load();

    switch (desc.scheme) {
        case ParticleScheme::full_f:
            ptl.psd        = { 1, f0(ptl), g0(ptl) };
            ptl.psd.weight = ptl.psd.real_f / ptl.psd.marker;
            break;
        case ParticleScheme::delta_f: {
            auto const scaling = uniform_real<494837>() * 2 - 1;
            ptl.psd            = { desc.initial_weight * scaling, f0(ptl), g0(ptl) };
            ptl.psd.real_f += ptl.psd.weight * ptl.psd.marker; // f = f_0 + w*g
            break;
        }
    }

    return ptl;
}
auto MaxwellianVDF::load() const -> Particle
{
    // position
    //
    CurviCoord const pos{ q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc) };

    // velocity in field-aligned frame (Hu et al., 2010, doi:10.1029/2009JA015158)
    //
    Real const phi1 = bit_reversed<3>() * 2 * M_PI;                               // [0, 2pi]
    Real const x1   = std::sqrt(-std::log(uniform_real<100>())) * std::sin(phi1); // v_para
    //
    Real const phi2 = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const tmp  = std::sqrt(-std::log(uniform_real<200>()) * T2OT1(pos));
    Real const x2   = std::cos(phi2) * tmp; // in-plane v_perp
    Real const x3   = std::sin(phi2) * tmp; // out-of-plane v_perp

    auto const vel = MFAVector{ x1, x2, x3 } * marker_vth1(pos);

    return { geomtr.mfa_to_cart(vel, pos), pos };
}
LIBPIC_NAMESPACE_END(1)
