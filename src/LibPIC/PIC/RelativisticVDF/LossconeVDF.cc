/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "LossconeVDF.h"
#include "../RandomReal.h"
#include "../VDFHelper.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <valarray>

LIBPIC_NAMESPACE_BEGIN(1)
RelativisticLossconeVDF::Params::Params(Real const losscone_beta, Real const vth1, Real const T2OT1) noexcept
: losscone_beta{ losscone_beta }
, vth1{ vth1 }
, vth1_cubed{ vth1 * vth1 * vth1 }
, xth2_square{ T2OT1 / (1 + losscone_beta) }
{
}
RelativisticLossconeVDF::RelativisticLossconeVDF(LossconePlasmaDesc const &desc, Geometry const &geo, Range const &domain_extent, Real c)
: RelativisticVDF{ geo, domain_extent, c }, desc{ desc }
{
    Real const losscone_beta = [beta = desc.losscone.beta] { // avoid beta == 1 && beta == 0
        if (beta < eps)
            return eps;
        if (Real const diff = beta - 1; std::abs(diff) < eps)
            return beta + std::copysign(eps, diff);
        return beta;
    }();
    //
    auto const vth1 = std::sqrt(desc.beta1) * c * std::abs(desc.Oc) / desc.op;
    m_physical_eq   = { losscone_beta, vth1, desc.T2_T1 };
    m_marker_eq     = { losscone_beta, vth1 * std::sqrt(desc.marker_temp_ratio), desc.T2_T1 };
    //
    m_N_extent.loc        = N_of_q1(domain_extent.min());
    m_N_extent.len        = N_of_q1(domain_extent.max()) - m_N_extent.loc;
    m_Nrefcell_div_Ntotal = (N_of_q1(+0.5) - N_of_q1(-0.5)) / m_N_extent.len;
    //
    m_q1_of_N = init_inverse_function_table(m_N_extent, domain_extent, [this](Real q1) {
        return N_of_q1(q1);
    });
}

auto RelativisticLossconeVDF::eta(CurviCoord const &pos) const noexcept -> Real
{
    auto const xth2_eq_square = m_physical_eq.xth2_square;
    //
    auto const cos = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    return 1 / (xth2_eq_square + (1 - xth2_eq_square) * cos * cos);
}
auto RelativisticLossconeVDF::eta_b(CurviCoord const &pos) const noexcept -> Real
{
    auto const beta_eq        = m_physical_eq.losscone_beta;
    auto const xth2_eq_square = m_physical_eq.xth2_square;
    //
    auto const cos = std::cos(geomtr.xi() * geomtr.D1() * pos.q1);
    auto const tmp = beta_eq * xth2_eq_square;
    return 1 / (tmp + (1 - tmp) * cos * cos);
}
auto RelativisticLossconeVDF::losscone_beta(CurviCoord const &pos) const noexcept -> Real
{
    auto const beta_eq = m_physical_eq.losscone_beta;
    auto const beta    = beta_eq * eta_b(pos) / eta(pos);
    // avoid beta == 1
    if (Real const diff = beta - 1; std::abs(diff) < eps)
        return beta + std::copysign(eps, diff);
    return beta;
}
auto RelativisticLossconeVDF::N_of_q1(Real const q1) const noexcept -> Real
{
    auto const beta_eq        = m_physical_eq.losscone_beta;
    auto const xth2_eq_square = m_physical_eq.xth2_square;
    if (geomtr.is_homogeneous()) {
        auto const xiD1q1 = geomtr.xi() * geomtr.D1() * q1;
        auto const tmp1   = 1 - (xth2_eq_square - 1) / 3 * xiD1q1 * xiD1q1;
        auto const tmp2   = 1 - (beta_eq * xth2_eq_square - 1) / 3 * xiD1q1 * xiD1q1;
        return q1 * (tmp1 - beta_eq * tmp2) / (1 - beta_eq);
    } else {
        auto const sqrt_beta_eq = std::sqrt(beta_eq);
        auto const xth2_eq      = std::sqrt(xth2_eq_square);
        auto const tan          = std::tan(geomtr.xi() * geomtr.D1() * q1);
        auto const tmp1         = std::atan(xth2_eq * tan) / (xth2_eq * geomtr.D1() * geomtr.xi());
        auto const tmp2         = std::atan(sqrt_beta_eq * xth2_eq * tan) / (sqrt_beta_eq * xth2_eq * geomtr.D1() * geomtr.xi());
        return (tmp1 - beta_eq * tmp2) / (1 - beta_eq);
    }
}
auto RelativisticLossconeVDF::q1_of_N(Real const N) const -> Real
{
    if (auto const q1 = linear_interp(m_q1_of_N, N))
        return *q1;
    throw std::out_of_range{ __PRETTY_FUNCTION__ };
}

auto RelativisticLossconeVDF::particle_flux_vector(CurviCoord const &pos) const -> FourMFAVector
{
    constexpr Real n0_eq   = 1;
    auto const     beta_eq = m_physical_eq.losscone_beta;
    auto const     n0      = n0_eq * (eta(pos) - beta_eq * eta_b(pos)) / (1 - beta_eq);
    return { n0 * c, {} };
}
auto RelativisticLossconeVDF::stress_energy_tensor(CurviCoord const &pos) const -> FourMFATensor
{
    auto const xth2_square   = this->xth2_square(pos);
    auto const losscone_beta = this->losscone_beta(pos);
    auto const vth1          = this->vth1(pos);
    auto const vth1_cubed    = this->vth1_cubed(pos);

    // define momentum space
    auto const u1max = vth1 * 4;
    auto const u1s   = [ulim = Range{ -1, 2 } * u1max] {
        std::array<Real, 2000> us{};
        std::iota(begin(us), end(us), long{});
        auto const du = ulim.len / us.size();
        for (auto &u : us) {
            (u *= du) += ulim.min() + du / 2;
        }
        return us;
    }();
    auto const du1 = u1s.at(1) - u1s.at(0);

    auto const u2max = vth1 * std::sqrt(xth2_square * std::max(Real{ 1 }, losscone_beta)) * 4.2;
    auto const u2s   = [ulim = Range{ 0, 1 } * u2max] {
        std::array<Real, 1500> us{};
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
        return (2 * M_PI * u2 * du2 * du1) * n0 * f_common(MFAVector{ u1, u2, 0 } / vth1, xth2_square, losscone_beta, vth1_cubed);
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

auto RelativisticLossconeVDF::f_common(MFAVector const &u0, Real const xth2_square, Real const losscone_beta, Real const denom) noexcept -> Real
{
    // note that the u0 is in co-moving frame and normalized by vth1
    //
    //                                  (exp(-(x2^2 + x3^2)/xth2^2) - exp(-(x2^2 + x3^2)/(β*xth2^2)))
    // f0(x1, x2, x3) = exp(-x1^2)/√π * -------------------------------------------------------------
    //                                                     (π * xth2^2 * (1 - β))
    //
    Real const f1 = std::exp(-u0.x * u0.x) * M_2_SQRTPI * .5;
    Real const f2 = [D     = 0,
                     b     = losscone_beta,
                     x2    = (u0.y * u0.y + u0.z * u0.z) / xth2_square,
                     denom = M_PI * xth2_square * (1 - losscone_beta)]() noexcept {
        return ((1 - D * b) * std::exp(-x2) - (1 - D) * std::exp(-x2 / b)) / denom;
    }();
    return (f1 * f2) / denom;
}
auto RelativisticLossconeVDF::f0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / vth1(pos), xth2_square(pos), losscone_beta(pos), vth1_cubed(pos));
}
auto RelativisticLossconeVDF::g0(FourCartVector const &gcgvel, CurviCoord const &pos) const noexcept -> Real
{
    // note that u = γ{v1, v2, v3} in lab frame, where γ = c/√(c^2 - v^2)
    auto const gcgv_mfa = geomtr.cart_to_mfa(gcgvel, pos);
    return Real{ this->n0(pos) } * f_common(gcgv_mfa.s / marker_vth1(pos), xth2_square(pos), losscone_beta(pos), marker_vth1_cubed(pos));
}

auto RelativisticLossconeVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto RelativisticLossconeVDF::impl_emit(Badge<Super>) const -> Particle
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
auto RelativisticLossconeVDF::load() const -> Particle
{
    // position
    //
    CurviCoord const pos{ q1_of_N(bit_reversed<2>() * m_N_extent.len + m_N_extent.loc) };

    // velocity in field-aligned, co-moving frame (Hu et al., 2010, doi:10.1029/2009JA015158)
    //
    Real const phi1 = bit_reversed<3>() * 2 * M_PI;                               // [0, 2pi]
    Real const u1   = std::sqrt(-std::log(uniform_real<100>())) * std::sin(phi1); // v_para
    //
    Real const phi2 = bit_reversed<5>() * 2 * M_PI; // [0, 2pi]
    Real const tmp  = RejectionSampler{ losscone_beta(pos) }.sample() * std::sqrt(xth2_square(pos));
    Real const u2   = std::cos(phi2) * tmp; // in-plane γv_perp
    Real const u3   = std::sin(phi2) * tmp; // out-of-plane γv_perp

    // boost from particle reference frame to co-moving frame
    auto const gcgv_mfa = lorentz_boost<-1>(FourMFAVector{ c, {} }, MFAVector{ u1, u2, u3 } * (marker_vth1(pos) / c));

    return { geomtr.mfa_to_cart(gcgv_mfa, pos), pos };
}

// MARK: - RejectionSampler
//
RelativisticLossconeVDF::RejectionSampler::RejectionSampler(Real const beta /*must not be 1*/)
: beta{ beta }
{
    if (std::abs(1 - Delta) < eps) { // Δ == 1
        alpha = 1;
        M     = 1;
    } else { // Δ != 1
        alpha               = (beta < 1 ? 1 : beta) + a_offset;
        auto const eval_xpk = [D = Delta, b = beta, a = alpha] {
            Real const det = -b / (1 - b) * std::log(((a - 1) * (1 - D * b) * b) / ((a - b) * (1 - D)));
            return std::isfinite(det) && det > 0 ? std::sqrt(det) : 0;
        };
        Real const xpk = std::abs(1 - Delta * beta) < eps ? 0 : eval_xpk();
        M              = fOg(xpk);
    }
    if (!std::isfinite(M))
        throw std::runtime_error{ __PRETTY_FUNCTION__ };
}
auto RelativisticLossconeVDF::RejectionSampler::fOg(const Real x) const noexcept -> Real
{
    using std::exp;
    Real const x2 = x * x;
    Real const f  = ((1 - Delta * beta) * exp(-x2) - (1 - Delta) * exp(-x2 / beta)) / (1 - beta);
    Real const g  = exp(-x2 / alpha) / alpha;
    return f / g; // ratio of the target distribution to proposed distribution
}
auto RelativisticLossconeVDF::RejectionSampler::sample() const noexcept -> Real
{
    auto const vote = [this](Real const proposal) noexcept {
        Real const jury = uniform_real<300>() * M;
        return jury <= fOg(proposal);
    };
    auto const proposed = [a = this->alpha]() noexcept {
        return std::sqrt(-std::log(uniform_real<200>()) * a);
    };
    //
    Real sample;
    while (!vote(sample = proposed())) {}
    return sample;
}
LIBPIC_NAMESPACE_END(1)
