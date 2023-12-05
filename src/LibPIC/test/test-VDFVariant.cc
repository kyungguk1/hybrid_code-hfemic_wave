/*
 * Copyright (c) 2021-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Misc/Faddeeva.hh>
#include <PIC/VDFVariant.h>
#include <algorithm>
#include <cmath>

namespace {
[[nodiscard]] bool operator==(Scalar const &a, Scalar const &b) noexcept
{
    return *a == Approx{ *b }.margin(1e-15);
}
template <class T, class U>
[[nodiscard]] bool operator==(Detail::VectorTemplate<T, double> const &a, Detail::VectorTemplate<U, double> const &b) noexcept
{
    return a.x == Approx{ b.x }.margin(1e-15)
        && a.y == Approx{ b.y }.margin(1e-15)
        && a.z == Approx{ b.z }.margin(1e-15);
}
template <class T1, class T2, class U1, class U2>
[[nodiscard]] bool operator==(Detail::TensorTemplate<T1, T2> const &a, Detail::TensorTemplate<U1, U2> const &b) noexcept
{
    return a.lo() == b.lo() && a.hi() == b.hi();
}
[[nodiscard]] bool operator==(CurviCoord const &a, CurviCoord const &b) noexcept
{
    return a.q1 == b.q1;
}
} // namespace
using ::operator==;

TEST_CASE("Test LibPIC::VDFVariant::TestParticleVDF", "[LibPIC::VDFVariant::TestParticleVDF]")
{
    Real const O0 = 1., op = 4 * O0, c = op;
    Real const xi = 0, D1 = 1;
    long const q1min = -7, q1max = 15;
    auto const geo   = Geometry{ xi, D1, O0 };
    auto const Nptls = 2;
    auto const desc  = TestParticleDesc<Nptls>{
        { -O0, op },
        { MFAVector{ 1, 2, 3 }, { 3, 4, 5 } },
        { CurviCoord{ q1min }, CurviCoord{ q1max } }
    };
    auto const vdf = *VDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        CHECK(vdf.n0(pos) == Scalar{ 0 });
        CHECK(geo.cart_to_mfa(vdf.nV0(pos), pos) == Vector{ 0, 0, 0 });
        CHECK(geo.cart_to_mfa(vdf.nvv0(pos), pos) == Tensor{ 0, 0, 0, 0, 0, 0 });
    }

    // sampling
    auto const n_samples = Nptls + 1;
    auto const particles = vdf.emit(n_samples);

    for (unsigned i = 0; i < Nptls; ++i) {
        auto const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == 0);
        REQUIRE(ptl.psd.real_f == 0);
        REQUIRE(ptl.psd.marker == 1);

        REQUIRE(ptl.pos == desc.pos[i]);
        REQUIRE(geo.cart_to_mfa(ptl.vel, ptl.pos) == desc.vel[i]);

        REQUIRE(vdf.real_f0(ptl) == 0);
    }
    {
        auto const &ptl = particles.back();

        REQUIRE(std::isnan(ptl.vel.x));
        REQUIRE(std::isnan(ptl.vel.y));
        REQUIRE(std::isnan(ptl.vel.z));
        REQUIRE(std::isnan(ptl.pos.q1));
    }
}

TEST_CASE("Test LibPIC::VDFVariant::MaxwellianVDF", "[LibPIC::VDFVariant::MaxwellianVDF]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = .1, T2OT1 = 5.35;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1, T2OT1);
    auto const vdf     = *VDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = MaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{};
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2,
            desc.beta1 / 2 * desc.T2_T1,
            desc.beta1 / 2 * desc.T2_T1,
            0,
            0,
            0
        };

        CHECK(vdf.n0(pos) == Scalar{ n0_ref });
        CHECK(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        CHECK(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    std::for_each_n(begin(particles), 100, [&](Particle const &ptl) {
        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-15));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-15));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-15));
    });
}

TEST_CASE("Test LibPIC::VDFVariant::LossconeVDF", "[LibPIC::VDFVariant::LossconeVDF]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 5.35;
    Real const xi = 0, D1 = 1.87, losscone_beta = 0.9, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1);
    auto const desc    = LossconePlasmaDesc({ losscone_beta }, BiMaxPlasmaDesc{ kinetic, beta1, T2OT1 });
    auto const vdf     = *VDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = LossconeVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = LossconePlasmaDesc({ losscone_beta }, { { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1 / (1 + losscone_beta));
    auto const g_vdf  = LossconeVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{};
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2,
            desc.beta1 / 2 * desc.T2_T1,
            desc.beta1 / 2 * desc.T2_T1,
            0,
            0,
            0
        };

        CHECK(vdf.n0(pos) == Scalar{ n0_ref });
        CHECK(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        CHECK(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    std::for_each_n(begin(particles), 100, [&](Particle const &ptl) {
        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-15));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
    });
}

TEST_CASE("Test LibPIC::VDFVariant::PartialShellVDF", "[LibPIC::VDFVariant::PartialShellVDF]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 30;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = *VDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = PartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref  = 1;
        auto const nV0_ref = Vector{};
        auto const nT_ref  = [xs = desc.vs / std::sqrt(beta)] {
            auto const xs2 = xs * xs;
            auto const Ab  = .5 * (xs * std::exp(-xs2) + 2 / M_2_SQRTPI * (0.5 + xs2) * std::erfc(-xs));
            return .5 / Ab * (xs * (2.5 + xs2) * std::exp(-xs2) + 2 / M_2_SQRTPI * (0.75 + xs2 * (3 + xs2)) * std::erfc(-xs));
        }() * n0_ref * beta;
        auto const nvv0_ref = Tensor{
            nT_ref / (3 + zeta),
            nT_ref / (3 + zeta) * (2 + zeta) * .5,
            nT_ref / (3 + zeta) * (2 + zeta) * .5,
            0,
            0,
            0
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    std::for_each_n(begin(particles), 100, [&](Particle const &ptl) {
        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-15));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
    });
}

TEST_CASE("Test LibPIC::VDFVariant::CounterBeamVDF", "[LibPIC::VDFVariant::CounterBeamVDF]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.1, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = *VDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = CounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta = 1;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const nu       = std::sqrt(geo.Bmag_div_B0(pos)) * nu0;
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const T1       = T * nu / 2 * (1 / Faddeeva::Dawson(1 / nu) - nu);
        auto const T2       = 0.5 * (T - T1);
        auto const nvv0_ref = Tensor{
            T1 * eta,
            T2 * eta,
            T2 * eta,
            0,
            0,
            0
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    std::for_each_n(begin(particles), 100, [&](Particle const &ptl) {
        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-15));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
    });
}
