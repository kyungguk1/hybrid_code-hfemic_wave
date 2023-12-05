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
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <valarray>

LIBPIC_NAMESPACE_BEGIN(1)
Real RelativisticCounterBeamVDF::Params::Bnu(Real const nu) noexcept
{
    return 2 * nu * Faddeeva::Dawson(1 / nu);
}
RelativisticCounterBeamVDF::Params::Params(Real const vth, Real const vs) noexcept
: vth{ vth }
, vth_cubed{ vth * vth * vth }
, xs{ vs / vth }
{
    Ab = [xs = this->xs] {
        auto const first  = xs * std::exp(-xs * xs);
        auto const second = 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs);
        return .5 * (first + second);
    }();
}
RelativisticCounterBeamVDF::RelativisticCounterBeamVDF(CounterBeamPlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real const c)
: RelativisticVDF{ geo, domain_extent, c }, desc{ desc }, nu0{ desc.nu }
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
Real RelativisticCounterBeamVDF::Fv_of_x(Real const v_by_vth) const noexcept
{
    auto const xs = m_marker.xs;
    auto const t  = v_by_vth - xs;
    return -(t + 2 * xs) * std::exp(-t * t) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erf(t);
}
Real RelativisticCounterBeamVDF::N_of_q1(Real const q1) const
{
    return integrate_dN(q1, [this](Real const q1) {
        return eta(CurviCoord{ q1 });
    });
}
Real RelativisticCounterBeamVDF::q1_of_N(Real const N) const
{
    if (auto const q1 = linear_interp(m_q1_of_N, N))
        return *q1;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
Real RelativisticCounterBeamVDF::x_of_Fv(Real const Fv) const
{
    if (auto const x = linear_interp(m_x_of_Fv, Fv))
        return *x;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}
Real RelativisticCounterBeamVDF::nu(CurviCoord const &pos) const noexcept
{
    auto const BOB0 = geomtr.Bmag_div_B0(pos);
    auto const nu   = nu0 * std::sqrt(BOB0);
    return nu;
}
Real RelativisticCounterBeamVDF::eta(CurviCoord const &pos) const noexcept
{
    return m_physical.Bnu(nu(pos)) / m_physical.Bnu(nu0);
}
auto RelativisticCounterBeamVDF::particle_flux_vector(CurviCoord const &pos) const -> FourMFAVector
{
    constexpr Real n0_eq = 1;
    auto const     n0    = n0_eq * eta(pos);
    return { n0 * c, {} };
}
auto RelativisticCounterBeamVDF::stress_energy_tensor(CurviCoord const &pos) const -> FourMFATensor
{
    auto const  nu_max    = std::max(nu(CurviCoord{ domain_extent.min() }),
                                     nu(CurviCoord{ domain_extent.max() }));
    auto const  nu        = this->nu(pos);
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

    // NOTE: here only consider first quadrant; be sure to multiply by 2 at the end
    auto const alim = [nu_max] {
        auto const a_max = std::asin(std::min(1.0, threshold_factor * nu_max));
        return Range{ 0, a_max };
    }();
    auto const alphas = linspace(alim);
    auto const dalpha = alphas.at(2) - alphas.at(1);

    // weight in the integrand
    // NOTE: the factor 2 in n0 is to account for the second quadrant in pitch angle space
    auto const n0     = 2 * *particle_flux_vector(pos).t / c;
    auto const weight = [&, du = duel, da = dalpha](Real const u, Real const a) {
        auto const u1 = u * std::cos(a);
        auto const u2 = u * std::sin(a);
        return n0 * (2 * M_PI * (u * u * std::sin(a)) * du * da)
             * f_common(MFAVector{ u1, u2, 0 } / vth, nu, shell, vth_cubed);
    };

    // evaluate integrand
    // 1. energy density       : ∫c    γ0c f0 du0
    // 2. c * momentum density : ∫c     u0 f0 du0, which is 0 in this case due to symmetry
    // 3. momentum flux        : ∫u0/γ0 u0 f0 du0
    auto const stress_energy = [&weight, c2 = c2](Real const u, Real const a) {
        auto const gamma = std::sqrt(1 + (u * u) / c2);
        auto const P1    = std::pow(u * std::cos(a), 2) / gamma;
        auto const P2    = .5 * std::pow(u * std::sin(a), 2) / gamma;
        return FourMFATensor{ gamma * c2, {}, { P1, P2, P2, 0, 0, 0 } } * weight(u, a);
    };
    auto const inner_loop = [&stress_energy, &alphas](Real const u) {
        std::valarray<FourMFATensor> integrand(alphas.size());
        std::transform(begin(alphas), end(alphas), begin(integrand), [&](Real const a) {
            return stress_energy(u, a);
        });
        return integrand.sum();
    };
    auto const outer_loop = [&inner_loop, &uels] {
        std::valarray<FourMFATensor> integrand(uels.size());
        std::transform(begin(uels), end(uels), begin(integrand), [&](Real const u) {
            return inner_loop(u);
        });
        return integrand.sum();
    };

    return outer_loop();
}

auto RelativisticCounterBeamVDF::f_common(MFAVector const &u0_by_vth, Real nu, Params const &shell, Real const denom) noexcept -> Real
{
    // note that the u0 is in co-moving frame and normalized by vth
    //
    // f0(x1, x2, x3) = exp(-(x - xs)^2)*exp(-sin^2α/ν^2)/(2π θ^3 A(xs) B(ν)),
    //
    auto const x    = std::sqrt(dot(u0_by_vth, u0_by_vth));
    auto const t    = x - shell.xs;
    Real const fv   = std::exp(-t * t) / shell.Ab;
    auto const cosa = u0_by_vth.x / x;
    Real const fa   = std::exp(-((1 - cosa) * (1 + cosa)) / (nu * nu)) / shell.Bnu(nu);
    return fv * fa / (2 * M_PI * denom);
}
auto RelativisticCounterBeamVDF::f0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / m_physical.vth, nu(pos), m_physical, m_physical.vth_cubed);
}
auto RelativisticCounterBeamVDF::g0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / m_marker.vth, nu(pos), m_marker, m_marker.vth_cubed);
}

auto RelativisticCounterBeamVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto RelativisticCounterBeamVDF::impl_emit(Badge<Super>) const -> Particle
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
auto RelativisticCounterBeamVDF::load() const -> Particle
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
    Real const u_vth = x_of_Fv(uniform_real<100>() * m_Fv_extent.len + m_Fv_extent.loc);
    Real const u1    = std::cos(alpha) * u_vth;
    Real const tmp   = std::sin(alpha) * u_vth;
    Real const u2    = std::cos(ph) * tmp;
    Real const u3    = std::sin(ph) * tmp;

    // boost from particle reference frame to co-moving frame
    auto const gcgv_mfa = lorentz_boost<-1>(FourMFAVector{ c, {} }, MFAVector{ u1, u2, u3 } * (m_marker.vth / c));

    return { geomtr.mfa_to_cart(gcgv_mfa, pos), pos };
}
RelativisticCounterBeamVDF::RejectionSampling::RejectionSampling(Real const nu) noexcept
: nu2{ nu * nu }
{
    a_max  = std::asin(std::min(1.0, threshold_factor * nu));
    f_peak = fa(std::asin(std::min(1.0, nu * M_SQRT1_2)));
}
Real RelativisticCounterBeamVDF::RejectionSampling::fa(Real const a) const noexcept
{
    auto const sina = std::sin(a);
    return std::exp(-sina * sina / nu2) * sina;
}
Real RelativisticCounterBeamVDF::RejectionSampling::sample() && noexcept
{
    auto const alpha = draw(); // 0 < alpha < π/2

    // coin flip to extend it to the second quadrant
    auto const coin = uniform_real<400>() - 0.5; // -0.5 < coin < 0.5
    return std::ceil(coin) * M_PI - std::copysign(alpha, coin);
}
Real RelativisticCounterBeamVDF::RejectionSampling::draw() const & noexcept
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
