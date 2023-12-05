/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "PartialShellVDF.h"
#include "../RandomReal.h"
#include "../VDFHelper.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
PartialShellVDF::Params::Params(Real const vth, unsigned const zeta, Real const vs) noexcept
: zeta{ zeta }, vth{ vth }, vth_cubed{ vth * vth * vth }, xs{ vs / vth }
{
    Ab = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
    Bz = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
    auto const T_vth2
        = .5 / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (0.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
    T1_by_vth2 = T_vth2 / (3 + zeta);
}
PartialShellVDF::PartialShellVDF(PartialShellPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c)
: VDF{ geo, domain_extent }, desc{ desc }
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

auto PartialShellVDF::eta(CurviCoord const &pos) const noexcept -> Real
{
    if (desc.zeta == 0)
        return 1;
    auto const cos = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    return std::pow(cos, desc.zeta);
}
auto PartialShellVDF::N_of_q1(Real const q1) const noexcept -> Real
{
    auto const scaling  = geomtr.xi();
    auto const argument = geomtr.D1() * q1;
    if (geomtr.is_homogeneous())
        return int_cos_zeta<true>(desc.zeta, scaling, argument);
    else
        return int_cos_zeta<false>(desc.zeta, scaling, argument);
}
auto PartialShellVDF::Fa_of_a(Real const alpha) const noexcept -> Real
{
    constexpr auto scaling  = 1;
    auto const     argument = alpha - M_PI_2;
    return int_cos_zeta<false>(desc.zeta + 1, scaling, argument);
}
auto PartialShellVDF::Fv_of_x(Real const v_by_vth) const noexcept -> Real
{
    auto const xs = m_marker.xs;
    auto const t  = v_by_vth - xs;
    return -(t + 2 * xs) * std::exp(-t * t) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erf(t);
}
auto PartialShellVDF::q1_of_N(Real const N) const -> Real
{
    if (auto const q1 = linear_interp(m_q1_of_N, N))
        return *q1;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
auto PartialShellVDF::x_of_Fv(Real const Fv) const -> Real
{
    if (auto const x = linear_interp(m_x_of_Fv, Fv))
        return *x;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
auto PartialShellVDF::a_of_Fa(Real const Fa) const -> Real
{
    if (auto const a = linear_interp(m_a_of_Fa, Fa))
        return *a;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}

auto PartialShellVDF::f_common(MFAVector const &v_by_vth, Params const &shell, Real const denom) noexcept -> Real
{
    // note that vel = {v1, v2, v3}/vth1
    // f(x1, x2, x3) = exp(-(x - xs)^2)*sin^ζ(α)/(2π θ^3 A(xs) B(ζ))
    //
    auto const x  = std::sqrt(dot(v_by_vth, v_by_vth));
    auto const t  = x - shell.xs;
    Real const fv = std::exp(-t * t) / shell.Ab;
    auto const u  = v_by_vth.x / x; // cos α
    Real const fa = (shell.zeta == 0 ? 1 : std::pow((1 - u) * (1 + u), .5 * shell.zeta)) / shell.Bz;
    return .5 * fv * fa / (M_PI * denom);
}
auto PartialShellVDF::f0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    return Real{ this->n0(pos) } * f_common(geomtr.cart_to_mfa(vel, pos) / m_physical.vth, m_physical, m_physical.vth_cubed);
}
auto PartialShellVDF::g0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    return Real{ this->n0(pos) } * f_common(geomtr.cart_to_mfa(vel, pos) / m_marker.vth, m_marker, m_marker.vth_cubed);
}

auto PartialShellVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto PartialShellVDF::impl_emit(Badge<Super>) const -> Particle
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
auto PartialShellVDF::load() const -> Particle
{
    // position
    //
    CurviCoord const pos{ q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc) };

    // velocity in field-aligned frame
    //
    Real const ph    = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const alpha = a_of_Fa(bit_reversed<3>() * m_Fa_extent.len + m_Fa_extent.loc);
    Real const v_vth = x_of_Fv(uniform_real<100>() * m_Fv_extent.len + m_Fv_extent.loc);
    Real const x1    = std::cos(alpha) * v_vth;
    Real const tmp   = std::sin(alpha) * v_vth;
    Real const x2    = std::cos(ph) * tmp;
    Real const x3    = std::sin(ph) * tmp;

    auto const vel = MFAVector{ x1, x2, x3 } * m_marker.vth;

    return { geomtr.mfa_to_cart(vel, pos), pos };
}
LIBPIC_NAMESPACE_END(1)
