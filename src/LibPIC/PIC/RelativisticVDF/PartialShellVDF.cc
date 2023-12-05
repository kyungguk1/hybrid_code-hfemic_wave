/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "PartialShellVDF.h"
#include "../RandomReal.h"
#include "../VDFHelper.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <valarray>

LIBPIC_NAMESPACE_BEGIN(1)
RelativisticPartialShellVDF::Params::Params(Real const vth, unsigned const zeta, Real const vs) noexcept
: zeta{ zeta }, vth{ vth }, vth_cubed{ vth * vth * vth }, xs{ vs / vth }
{
    Ab = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
    Bz = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
}
RelativisticPartialShellVDF::RelativisticPartialShellVDF(PartialShellPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c)
: RelativisticVDF{ geo, domain_extent, c }, desc{ desc }
{
    auto const vth = std::sqrt(desc.beta) * c * std::abs(desc.Oc) / desc.op;
    m_physical     = { vth, desc.zeta, desc.vs };
    m_marker       = { vth * std::sqrt(desc.marker_temp_ratio), desc.zeta, desc.vs };
    //
    { // initialize q1 integral table
        m_N_extent.loc        = N_of_q1(domain_extent.min());
        m_N_extent.len        = N_of_q1(domain_extent.max()) - m_N_extent.loc;
        m_Nrefcell_div_Ntotal = (N_of_q1(+0.5) - N_of_q1(-0.5)) / m_N_extent.len;
        //
        m_q1_of_N = init_inverse_function_table(m_N_extent, domain_extent, [this](Real q1) {
            return N_of_q1(q1);
        });
    }
    { // initialize velocity integral table
        constexpr auto t_max    = 5;
        auto const     xs       = m_marker.xs;
        auto const     t_min    = -(xs < t_max ? xs : t_max);
        Range const    x_extent = { t_min + xs, t_max - t_min };
        // provisional extent
        m_Fv_extent.loc = Fv_of_x(x_extent.min());
        m_Fv_extent.len = Fv_of_x(x_extent.max()) - m_Fv_extent.loc;
        m_x_of_Fv       = init_inverse_function_table(m_Fv_extent, x_extent, [this](Real x) {
            return Fv_of_x(x);
        });
        // FIXME: Chopping the head and tail off is a hackish solution of fixing anomalous particle initialization close to the boundaries.
        m_x_of_Fv.erase(m_Fv_extent.min());
        m_x_of_Fv.erase(m_Fv_extent.max());
        // finalized extent
        m_Fv_extent.loc = m_x_of_Fv.begin()->first;
        m_Fv_extent.len = m_x_of_Fv.rbegin()->first - m_Fv_extent.loc;
    }
    { // initialize pitch angle integral table
        constexpr auto accuracy_goal = 10;
        auto const     ph_max        = std::acos(std::pow(10, -accuracy_goal / Real(m_marker.zeta + 1)));
        auto const     ph_min        = -ph_max;
        Range const    a_extent      = { ph_min + M_PI_2, ph_max - ph_min };
        // provisional extent
        m_Fa_extent.loc = Fa_of_a(a_extent.min());
        m_Fa_extent.len = Fa_of_a(a_extent.max()) - m_Fa_extent.loc;
        m_a_of_Fa       = init_inverse_function_table(m_Fa_extent, a_extent, [this](Real a) {
            return Fa_of_a(a);
        });
        // FIXME: Chopping the head and tail off is a hackish solution of fixing anomalous particle initialization close to the boundaries.
        m_a_of_Fa.erase(m_Fa_extent.min());
        m_a_of_Fa.erase(m_Fa_extent.max());
        // finalized extent
        m_Fa_extent.loc = m_a_of_Fa.begin()->first;
        m_Fa_extent.len = m_a_of_Fa.rbegin()->first - m_Fa_extent.loc;
    }
}

auto RelativisticPartialShellVDF::eta(CurviCoord const &pos) const noexcept -> Real
{
    if (desc.zeta == 0)
        return 1;
    auto const cos = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    return std::pow(cos, desc.zeta);
}
auto RelativisticPartialShellVDF::N_of_q1(Real const q1) const noexcept -> Real
{
    auto const scaling  = geomtr.xi();
    auto const argument = geomtr.D1() * q1;
    if (geomtr.is_homogeneous())
        return int_cos_zeta<true>(desc.zeta, scaling, argument);
    else
        return int_cos_zeta<false>(desc.zeta, scaling, argument);
}
auto RelativisticPartialShellVDF::Fa_of_a(Real const alpha) const noexcept -> Real
{
    constexpr auto scaling  = 1;
    auto const     argument = alpha - M_PI_2;
    return int_cos_zeta<false>(desc.zeta + 1, scaling, argument);
}
auto RelativisticPartialShellVDF::Fv_of_x(Real const v_by_vth) const noexcept -> Real
{
    auto const xs = m_marker.xs;
    auto const t  = v_by_vth - xs;
    return -(t + 2 * xs) * std::exp(-t * t) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erf(t);
}
auto RelativisticPartialShellVDF::q1_of_N(Real const N) const -> Real
{
    if (auto const q1 = linear_interp(m_q1_of_N, N))
        return *q1;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
auto RelativisticPartialShellVDF::x_of_Fv(Real const Fv) const -> Real
{
    if (auto const x = linear_interp(m_x_of_Fv, Fv))
        return *x;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
auto RelativisticPartialShellVDF::a_of_Fa(Real const Fa) const -> Real
{
    if (auto const a = linear_interp(m_a_of_Fa, Fa))
        return *a;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}

auto RelativisticPartialShellVDF::particle_flux_vector(CurviCoord const &pos) const -> FourMFAVector
{
    constexpr Real n0_eq = 1;
    auto const     n0    = n0_eq * eta(pos);
    return { n0 * c, {} };
}
auto RelativisticPartialShellVDF::stress_energy_tensor(CurviCoord const &pos) const -> FourMFATensor
{
    auto const  zeta      = m_physical.zeta;
    auto const  vth       = m_physical.vth;
    auto const  vth_cubed = m_physical.vth_cubed;
    auto const  xs        = m_physical.xs;
    auto const &shell     = m_physical;

    // define momentum space
    auto const linspace = [](Range const &ulim) {
        std::array<Real, 2000> us{};
        std::iota(begin(us), end(us), long{});
        auto const du = ulim.len / us.size();
        for (auto &u : us) {
            (u *= du) += ulim.min() + du / 2;
        }
        return us;
    };

    auto const ulim = [vth, xs] {
        constexpr auto t_max = 5;
        auto const     t_min = -(xs < t_max ? xs : t_max);
        return Range{ t_min + xs, t_max - t_min } * vth;
    }();
    auto const uels = linspace(ulim);
    auto const duel = uels.at(2) - uels.at(1);

    auto const alim = [zeta] {
        constexpr auto accuracy_goal = 10;
        auto const     ph_max        = std::acos(std::pow(10, -accuracy_goal / Real(zeta + 1)));
        auto const     ph_min        = -ph_max;
        return Range{ ph_min + M_PI_2, ph_max - ph_min };
    }();
    auto const alphas = linspace(alim);
    auto const dalpha = alphas.at(2) - alphas.at(1);

    // weight in the integrand
    auto const n0     = *particle_flux_vector(pos).t / c;
    auto const weight = [&, du = duel, da = dalpha](Real const u, Real const a) {
        auto const u1 = u * std::cos(a);
        auto const u2 = u * std::sin(a);
        return (2 * M_PI * (u * u * std::sin(a)) * du * da) * n0 * f_common(MFAVector{ u1, u2, 0 } / vth, shell, vth_cubed);
    };

    // evaluate integrand
    // 1. energy density       : ∫c    γ0c f0 du0
    // 2. c * momentum density : ∫c     u0 f0 du0, which is 0 in this case due to symmetry
    // 3. momentum flux        : ∫u0/γ0 u0 f0 du0
    auto const inner_loop = [&](Real const u) {
        std::valarray<FourMFATensor> integrand(alphas.size());
        std::transform(begin(alphas), end(alphas), begin(integrand), [&](Real const a) {
            auto const gamma = std::sqrt(1 + (u * u) / c2);
            auto const P1    = std::pow(u * std::cos(a), 2) / gamma;
            auto const P2    = .5 * std::pow(u * std::sin(a), 2) / gamma;
            return FourMFATensor{ gamma * c2, {}, { P1, P2, P2, 0, 0, 0 } } * weight(u, a);
        });
        return integrand.sum();
    };
    auto const outer_loop = [inner_loop, &uels] {
        std::valarray<FourMFATensor> integrand(uels.size());
        std::transform(begin(uels), end(uels), begin(integrand), [&](Real const u) {
            return inner_loop(u);
        });
        return integrand.sum();
    };

    return outer_loop();
}

auto RelativisticPartialShellVDF::f_common(MFAVector const &u0_by_vth, Params const &shell, Real const denom) noexcept -> Real
{
    // note that the u0 is in co-moving frame and normalized by vth
    //
    // f0(x1, x2, x3) = exp(-(x - xs)^2)*sin^ζ(α)/(2π θ^3 A(xs) B(ζ))
    //
    auto const x  = std::sqrt(dot(u0_by_vth, u0_by_vth));
    auto const t  = x - shell.xs;
    Real const fv = std::exp(-t * t) / shell.Ab;
    auto const mu = u0_by_vth.x / x;
    Real const fa = (shell.zeta == 0 ? 1 : std::pow((1 - mu) * (1 + mu), .5 * shell.zeta)) / shell.Bz;
    return .5 * fv * fa / (M_PI * denom);
}
auto RelativisticPartialShellVDF::f0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / m_physical.vth, m_physical, m_physical.vth_cubed);
}
auto RelativisticPartialShellVDF::g0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / m_marker.vth, m_marker, m_marker.vth_cubed);
}

auto RelativisticPartialShellVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto RelativisticPartialShellVDF::impl_emit(Badge<Super>) const -> Particle
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
auto RelativisticPartialShellVDF::load() const -> Particle
{
    // position
    //
    CurviCoord const pos{ q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc) };

    // velocity in field-aligned frame
    //
    Real const ph    = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const alpha = a_of_Fa(bit_reversed<3>() * m_Fa_extent.len + m_Fa_extent.loc);
    Real const u_vth = x_of_Fv(uniform_real<100>() * m_Fv_extent.len + m_Fv_extent.loc);
    Real const u1    = std::cos(alpha) * u_vth;
    Real const tmp   = std::sin(alpha) * u_vth;
    Real const u2    = std::cos(ph) * tmp;
    Real const u3    = std::sin(ph) * tmp;

    // boost from particle reference frame to co-moving frame
    auto const gcgv_mfa = lorentz_boost<-1>(FourMFAVector{ c, {} }, MFAVector{ u1, u2, u3 } * (m_marker.vth / c));

    return { geomtr.mfa_to_cart(gcgv_mfa, pos), pos };
}
LIBPIC_NAMESPACE_END(1)
