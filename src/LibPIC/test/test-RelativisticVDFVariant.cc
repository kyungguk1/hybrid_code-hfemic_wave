/*
 * Copyright (c) 2021-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/RelativisticVDFVariant.h>
#include <algorithm>
#include <cmath>

namespace {
using Particle = RelativisticParticle;

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
template <class T1, class T2, class U1, class U2>
[[nodiscard]] bool operator==(Detail::FourVectorTemplate<T1, T2> const &a, Detail::FourVectorTemplate<U1, U2> const &b) noexcept
{
    return a.t == Approx{ b.t }.margin(1e-15) && a.s == b.s;
}
template <class T1, class T2, class T3, class U1, class U2, class U3>
[[nodiscard]] bool operator==(Detail::FourTensorTemplate<T1, T2, T3> const &a, Detail::FourTensorTemplate<U1, U2, U3> const &b) noexcept
{
    return a.tt == Approx{ b.tt }.margin(1e-15)
        && a.ts == b.ts
        && a.ss == b.ss;
}
[[nodiscard]] bool operator==(CurviCoord const &a, CurviCoord const &b) noexcept
{
    return a.q1 == b.q1;
}
} // namespace
using ::operator==;

TEST_CASE("Test LibPIC::RelativisticVDFVariant::TestParticleVDF", "[LibPIC::RelativisticVDFVariant::TestParticleVDF]")
{
    Real const O0 = 1., op = 10 * O0, c = op;
    Real const xi = 0, D1 = 1;
    long const q1min = -7, q1max = 15;
    auto const geo   = Geometry{ xi, D1, O0 };
    auto const Nptls = 2;
    auto const desc  = TestParticleDesc<Nptls>{
        { -O0, op },
        { MFAVector{ 1, 2, 3 }, { 3, 4, 5 } },
        { CurviCoord{ q1min }, CurviCoord{ q1max } }
    };
    auto const vdf = *RelativisticVDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        CHECK(vdf.n0(pos) == Scalar{ 0 });
        CHECK(geo.cart_to_mfa(vdf.nV0(pos), pos) == Vector{ 0, 0, 0 });
        CHECK(geo.cart_to_mfa(vdf.nuv0(pos), pos) == FourTensor{ 0, { 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } });
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
        REQUIRE(geo.cart_to_mfa(ptl.gcgvel, ptl.pos) == lorentz_boost<-1>(FourMFAVector{ c, {} }, desc.vel[i] / c, Real{ ptl.gcgvel.t } / c));

        REQUIRE(vdf.real_f0(ptl) == 0);
    }
    {
        auto const &ptl = particles.back();

        REQUIRE(std::isnan(*ptl.gcgvel.t));
        REQUIRE(std::isnan(ptl.gcgvel.s.x));
        REQUIRE(std::isnan(ptl.gcgvel.s.y));
        REQUIRE(std::isnan(ptl.gcgvel.s.z));
        REQUIRE(std::isnan(ptl.pos.q1));
    }
}

TEST_CASE("Test LibPIC::RelativisticVDFVariant::MaxwellianVDF", "[LibPIC::RelativisticVDFVariant::MaxwellianVDF]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 5.35;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1, T2OT1);
    auto const vdf     = *RelativisticVDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = RelativisticMaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{};
        auto const nuv0_ref = FourTensor{
            19.68844169847468,
            { 0, 0, 0 },
            { 0.6024514298876814, 2.913078061288704, 2.913078061288704, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));
    }
}

TEST_CASE("Test LibPIC::RelativisticVDFVariant::LossconeVDF", "[LibPIC::RelativisticVDFVariant::LossconeVDF]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35, beta_eq = .9;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = LossconePlasmaDesc({ beta_eq }, kinetic, beta1_eq, T2OT1_eq / (1 + beta_eq));
    auto const vdf     = *RelativisticVDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = RelativisticLossconeVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = RelativisticLossconeVDF(LossconePlasmaDesc{ { beta_eq }, g_desc }, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = FourTensor{
            19.786975759925791607,
            { 0, 0, 0 },
            { 0.59535600305974989421, 3.0518980566112574593, 3.0518980566112574593, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));
    }
}

TEST_CASE("Test LibPIC::RelativisticVDFVariant::PartialShellVDF", "[LibPIC::RelativisticVDFVariant::PartialShellVDF]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 30;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = *RelativisticVDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = RelativisticPartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = FourTensor{
            43.653319735303327320,
            { 0, 0, 0 },
            { 1.1441542043940300388, 18.306467234415908507, 18.306467234415908507, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));
    }
}

TEST_CASE("Test LibPIC::RelativisticVDFVariant::CounterBeamVDF", "[LibPIC::RelativisticVDFVariant::CounterBeamVDF]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.1, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = *RelativisticVDFVariant::make(desc, geo, Range{ q1min, q1max - q1min }, c);

    auto const f_vdf  = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);
    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = RelativisticCounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(kinetic) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = FourTensor{
            43.653351654602829512,
            { 0, 0, 0 },
            { 37.377608355377660132, 0.18975388435494586203, 0.18975388435494586203, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ f_vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ f_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ f_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));
    }
}
