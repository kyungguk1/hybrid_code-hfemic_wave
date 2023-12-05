/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "MaxwellianVDF.h"
#include "../RandomReal.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <valarray>

LIBPIC_NAMESPACE_BEGIN(1)
RelativisticMaxwellianVDF::Params::Params(Real const vth1, Real const T2OT1) noexcept
: vth1{ vth1 }
, T2OT1{ T2OT1 }
, sqrt_T2OT1{ std::sqrt(T2OT1) }
, vth1_square{ vth1 * vth1 }
, vth1_cubed{ vth1 * vth1 * vth1 }
{
}
RelativisticMaxwellianVDF::RelativisticMaxwellianVDF(BiMaxPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c)
: RelativisticVDF{ geo, domain_extent, c }, desc{ desc }
{ // parameter check is assumed to be done already
    auto const vth1 = std::sqrt(desc.beta1) * c * std::abs(desc.Oc) / desc.op;
    m_physical_eq   = { vth1, desc.T2_T1 };
    m_marker_eq     = { vth1 * std::sqrt(desc.marker_temp_ratio), desc.T2_T1 };
    //
    m_N_extent.loc        = N_of_q1(domain_extent.min());
    m_N_extent.len        = N_of_q1(domain_extent.max()) - m_N_extent.loc;
    m_Nrefcell_div_Ntotal = (N_of_q1(+0.5) - N_of_q1(-0.5)) / m_N_extent.len;
}

auto RelativisticMaxwellianVDF::eta(CurviCoord const &pos) const noexcept -> Real
{
    auto const T2OT1_eq = m_physical_eq.T2OT1;
    auto const cos      = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    return 1 / (T2OT1_eq + (1 - T2OT1_eq) * cos * cos);
}
auto RelativisticMaxwellianVDF::T2OT1(CurviCoord const &pos) const noexcept -> Real
{
    auto const T2OT1_eq = m_physical_eq.T2OT1;
    return T2OT1_eq * eta(pos);
}
auto RelativisticMaxwellianVDF::N_of_q1(Real const q1) const noexcept -> Real
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
auto RelativisticMaxwellianVDF::q1_of_N(Real const N) const noexcept -> Real
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

auto RelativisticMaxwellianVDF::particle_flux_vector(CurviCoord const &pos) const -> FourMFAVector
{
    constexpr Real n0_eq = 1;
    auto const     n0    = n0_eq * eta(pos);
    return { n0 * c, {} };
}
auto RelativisticMaxwellianVDF::stress_energy_tensor(CurviCoord const &pos) const -> FourMFATensor
{
    auto const T2OT1      = this->T2OT1(pos);
    auto const vth1       = this->vth1(pos);
    auto const vth1_cubed = this->vth1_cubed(pos);

    // define momentum space
    auto const u1lim = Range{ -1, 2 } * vth1 * 4;
    auto const u1s   = [&ulim = u1lim] {
        std::array<Real, 2000> us{};
        std::iota(begin(us), end(us), long{});
        auto const du = ulim.len / us.size();
        for (auto &u : us) {
            (u *= du) += ulim.min() + du / 2;
        }
        return us;
    }();
    auto const du1 = u1s.at(1) - u1s.at(0);

    auto const u2lim = Range{ 0, 1 } * vth1 * std::sqrt(T2OT1) * 4;
    auto const u2s   = [&ulim = u2lim] {
        std::array<Real, 1000> us{};
        std::iota(begin(us), end(us), long{});
        auto const du = ulim.len / us.size();
        for (auto &u : us) {
            (u *= du) += ulim.min() + du / 2;
        }
        return us;
    }();
    auto const du2 = u2s.at(1) - u2s.at(0);

    // weight in the integrand
    auto const n0     = *particle_flux_vector(pos).t / c;
    auto const weight = [&](Real const u1, Real const u2) {
        return (2 * M_PI * u2 * du2 * du1) * n0 * f_common(MFAVector{ u1, u2, 0 } / vth1, T2OT1, vth1_cubed);
    };

    // evaluate integrand
    // 1. energy density       : ∫c    γ0c f0 du0
    // 2. c * momentum density : ∫c     u0 f0 du0, which is 0 in this case due to symmetry
    // 3. momentum flux        : ∫u0/γ0 u0 f0 du0
    auto const inner_loop = [c2 = this->c2, &weight, &u2s](Real const u1) {
        std::valarray<FourMFATensor> integrand(u2s.size());
        std::transform(begin(u2s), end(u2s), begin(integrand), [&](Real const u2) {
            auto const gamma = std::sqrt(1 + (u1 * u1 + u2 * u2) / c2);
            auto const P1    = u1 * u1 / gamma;
            auto const P2    = .5 * u2 * u2 / gamma;
            return FourMFATensor{ gamma * c2, {}, { P1, P2, P2, 0, 0, 0 } } * weight(u1, u2);
        });
        return integrand.sum();
    };
    auto const outer_loop = [inner_loop, &u1s] {
        std::valarray<FourMFATensor> integrand(u1s.size());
        std::transform(begin(u1s), end(u1s), begin(integrand), [&](Real const u1) {
            return inner_loop(u1);
        });
        return integrand.sum();
    };

    return outer_loop();
}

auto RelativisticMaxwellianVDF::f_common(MFAVector const &u0, Real const T2OT1, Real const denom) noexcept -> Real
{
    // note that the u0 is in co-moving frame and normalized by vth1
    // f0(u1, u2, u3) = exp(-u1^2)/√π * exp(-(u2^2 + u3^2)/(T2/T1))/(π T2/T1)
    //
    Real const f1   = std::exp(-u0.x * u0.x) * M_2_SQRTPI * .5;
    Real const perp = u0.y * u0.y + u0.z * u0.z;
    Real const f2   = std::exp(-perp / T2OT1) / (M_PI * T2OT1);
    return (f1 * f2) / denom;
}
auto RelativisticMaxwellianVDF::f0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / vth1(pos), T2OT1(pos), vth1_cubed(pos));
}
auto RelativisticMaxwellianVDF::g0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / marker_vth1(pos), T2OT1(pos), marker_vth1_cubed(pos));
}

auto RelativisticMaxwellianVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto RelativisticMaxwellianVDF::impl_emit(Badge<Super>) const -> Particle
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
auto RelativisticMaxwellianVDF::load() const -> Particle
{
    // position
    //
    CurviCoord const pos{ q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc) };

    // velocity in field-aligned co-moving frame (Hu et al., 2010, doi:10.1029/2009JA015158)
    //
    Real const phi1 = bit_reversed<3>() * 2 * M_PI;                               // [0, 2pi]
    Real const u1   = std::sqrt(-std::log(uniform_real<100>())) * std::sin(phi1); // γv_para
    //
    Real const phi2 = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const tmp  = std::sqrt(-std::log(uniform_real<200>()) * T2OT1(pos));
    Real const u2   = std::cos(phi2) * tmp; // in-plane γv_perp
    Real const u3   = std::sin(phi2) * tmp; // out-of-plane γv_perp

    // boost from particle reference frame to co-moving frame
    auto const gcgv_mfa = lorentz_boost<-1>(FourMFAVector{ c, {} }, MFAVector{ u1, u2, u3 } * (marker_vth1(pos) / c));

    return { geomtr.mfa_to_cart(gcgv_mfa, pos), pos };
}
LIBPIC_NAMESPACE_END(1)
