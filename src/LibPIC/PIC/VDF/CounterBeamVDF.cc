/*
 * Copyright (c) 2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "CounterBeamVDF.h"
#include "../Misc/Faddeeva.hh"
#include "../RandomReal.h"
#include "../VDFHelper.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
Real CounterBeamVDF::Params::Bnu(Real const nu) noexcept
{
    return 2 * nu * Faddeeva::Dawson(1 / nu);
}
CounterBeamVDF::Params::Params(Real const vth, Real const vs) noexcept
: vth{ vth }
, vth_cubed{ vth * vth * vth }
, xs{ vs / vth }
{
    Ab = [xs = this->xs] {
        auto const first  = xs * std::exp(-xs * xs);
        auto const second = 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs);
        return .5 * (first + second);
    }();
    T_by_vth2 = [xs = this->xs, Ab = this->Ab] {
        auto const first  = xs * (2.5 + xs * xs) * std::exp(-xs * xs);
        auto const second = 2 / M_2_SQRTPI * (0.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs);
        return .5 / Ab * (first + second);
    }();
}
CounterBeamVDF::CounterBeamVDF(CounterBeamPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real const c)
: VDF{ geo, domain_extent }, desc{ desc }, nu0{ desc.nu }
{
    {
        auto const vth = std::sqrt(desc.beta) * c * std::abs(desc.Oc) / desc.op;
        m_physical     = { vth, desc.vs };
        m_marker       = { vth * std::sqrt(desc.marker_temp_ratio), desc.vs };
    }
    { // initialize q1(N) table
        m_N_extent.loc        = N_of_q1(domain_extent.min());
        m_N_extent.len        = N_of_q1(domain_extent.max()) - m_N_extent.loc;
        m_Nrefcell_div_Ntotal = (N_of_q1(+0.5) - N_of_q1(-0.5)) / m_N_extent.len;
        if (!std::isfinite(m_Nrefcell_div_Ntotal))
            throw std::domain_error{ std::string{ __PRETTY_FUNCTION__ } + " - not a number returned from `N_of_q1`" };

        m_q1_of_N = build_q1_of_N_interpolation_table(m_N_extent, domain_extent, [this](Real q1) {
            return eta(CurviCoord{ q1 });
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

        m_x_of_Fv = init_inverse_function_table(m_Fv_extent, x_extent, [this](Real x) {
            return Fv_of_x(x);
        });
        // FIXME: Chopping the head and tail off is a hackish solution of fixing anomalous particle initialization close to the boundaries.
        m_x_of_Fv.erase(m_Fv_extent.min());
        m_x_of_Fv.erase(m_Fv_extent.max());
        // finalized extent
        m_Fv_extent.loc = m_x_of_Fv.begin()->first;
        m_Fv_extent.len = m_x_of_Fv.rbegin()->first - m_Fv_extent.loc;
    }
}
Real CounterBeamVDF::Fv_of_x(Real const v_by_vth) const noexcept
{
    auto const xs = m_marker.xs;
    auto const t  = v_by_vth - xs;
    return -(t + 2 * xs) * std::exp(-t * t) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erf(t);
}
Real CounterBeamVDF::N_of_q1(Real const q1) const
{
    return integrate_dN(q1, [this](Real const q1) {
        return eta(CurviCoord{ q1 });
    });
}
Real CounterBeamVDF::q1_of_N(Real const N) const
{
    if (auto const q1 = linear_interp(m_q1_of_N, N))
        return *q1;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
Real CounterBeamVDF::x_of_Fv(Real const Fv) const
{
    if (auto const x = linear_interp(m_x_of_Fv, Fv))
        return *x;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
Real CounterBeamVDF::nu(CurviCoord const &pos) const noexcept
{
    auto const BOB0 = geomtr.Bmag_div_B0(pos);
    auto const nu   = nu0 * std::sqrt(BOB0);
    return nu;
}
Real CounterBeamVDF::eta(CurviCoord const &pos) const noexcept
{
    return m_physical.Bnu(nu(pos)) / m_physical.Bnu(nu0);
}
auto CounterBeamVDF::T1_and_T2_by_T(CurviCoord const &pos) const noexcept -> std::pair<Real, Real>
{
    auto const nu   = this->nu(pos);
    auto const T1OT = 0.5 * nu * (1 / Faddeeva::Dawson(1 / nu) - nu);
    auto const T2OT = 0.5 * (1 - T1OT);
    return std::make_pair(T1OT, T2OT);
}

auto CounterBeamVDF::f_common(MFAVector const &v_by_vth, Real nu, Params const &shell, Real const denom) noexcept -> Real
{
    // note that vel = {v1, v2, v3}/vth
    // f(x1, x2, x3) = exp(-(x - xs)^2)*exp(-sin^2α/ν^2)/(2π θ^3 A(xs) B(ν)),
    //
    auto const x    = std::sqrt(dot(v_by_vth, v_by_vth));
    auto const t    = x - shell.xs;
    Real const fv   = std::exp(-t * t) / shell.Ab;
    auto const cosa = v_by_vth.x / x;
    Real const fa   = std::exp(-((1 - cosa) * (1 + cosa)) / (nu * nu)) / shell.Bnu(nu);
    return fv * fa / (2 * M_PI * denom);
}
auto CounterBeamVDF::f0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    return Real{ this->n0(pos) } * f_common(geomtr.cart_to_mfa(vel, pos) / m_physical.vth, nu(pos), m_physical, m_physical.vth_cubed);
}
auto CounterBeamVDF::g0(CartVector const &vel, CurviCoord const &pos) const noexcept -> Real
{
    return Real{ this->n0(pos) } * f_common(geomtr.cart_to_mfa(vel, pos) / m_marker.vth, nu(pos), m_marker, m_marker.vth_cubed);
}

auto CounterBeamVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto CounterBeamVDF::impl_emit(Badge<Super>) const -> Particle
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
auto CounterBeamVDF::load() const -> Particle
{
    // position
    //
    auto const pos = [this] {
        Real q1;
        do {
            q1 = q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc);
        } while (!domain_extent.is_member(q1));
        return CurviCoord{ q1 };
    }();

    // velocity in field-aligned frame
    //
    Real const ph    = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const alpha = RejectionSampling{ nu(pos) }.sample();
    Real const v_vth = x_of_Fv(uniform_real<100>() * m_Fv_extent.len + m_Fv_extent.loc);
    Real const x1    = std::cos(alpha) * v_vth;
    Real const tmp   = std::sin(alpha) * v_vth;
    Real const x2    = std::cos(ph) * tmp;
    Real const x3    = std::sin(ph) * tmp;

    auto const vel = MFAVector{ x1, x2, x3 } * m_marker.vth;

    return { geomtr.mfa_to_cart(vel, pos), pos };
}
CounterBeamVDF::RejectionSampling::RejectionSampling(Real const nu) noexcept
: nu2{ nu * nu }
{
    a_max  = std::asin(std::min(1.0, threshold_factor * nu));
    f_peak = fa(std::asin(std::min(1.0, nu * M_SQRT1_2)));
}
Real CounterBeamVDF::RejectionSampling::fa(Real const a) const noexcept
{
    auto const sina = std::sin(a);
    return std::exp(-sina * sina / nu2) * sina;
}
Real CounterBeamVDF::RejectionSampling::sample() && noexcept
{
    auto const alpha = draw(); // 0 < alpha < π/2

    // coin flip to extend it to the second quadrant
    auto const coin = uniform_real<400>() - 0.5; // -0.5 < coin < 0.5
    return std::ceil(coin) * M_PI - std::copysign(alpha, coin);
}
Real CounterBeamVDF::RejectionSampling::draw() const & noexcept
{
    auto const vote = [this](Real const proposal) noexcept {
        Real const jury = uniform_real<300>() * f_peak;
        return jury <= fa(proposal);
    };
    constexpr auto proposed = [](Real max) noexcept {
        return uniform_real<200>() * max;
    };

    Real draw;
    while (!vote(draw = proposed(a_max))) {}
    return draw;
}
LIBPIC_NAMESPACE_END(1)
