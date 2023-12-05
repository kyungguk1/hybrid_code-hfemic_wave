/*
 * Copyright (c) 2021-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Misc/Faddeeva.hh>
#include <PIC/RelativisticVDF/CounterBeamVDF.h>
#include <PIC/RelativisticVDF/LossconeVDF.h>
#include <PIC/RelativisticVDF/MaxwellianVDF.h>
#include <PIC/RelativisticVDF/PartialShellVDF.h>
#include <PIC/RelativisticVDF/TestParticleVDF.h>
#include <PIC/UTL/println.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace {
constexpr bool dump_samples         = false;
constexpr bool enable_moment_checks = false;

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

TEST_CASE("Test LibPIC::RelativisticVDF::TestParticleVDF", "[LibPIC::RelativisticVDF::TestParticleVDF]")
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
    auto const vdf = RelativisticTestParticleVDF(desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    CHECK(vdf.initial_number_of_test_particles == Nptls);
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        CHECK(vdf.n0(pos) == Scalar{ 0 });
        CHECK(geo.cart_to_mfa(vdf.nV0(pos), pos) == MFAVector{ 0, 0, 0 });
        CHECK(geo.cart_to_mfa(vdf.nuv0(pos), pos) == FourMFATensor{ 0, { 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } });
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
        REQUIRE(vdf.g0(ptl) == 1);
    }
    {
        auto const &ptl = particles.back();

        REQUIRE(std::isnan(*ptl.gcgvel.t));
        REQUIRE(std::isnan(ptl.gcgvel.s.x));
        REQUIRE(std::isnan(ptl.gcgvel.s.y));
        REQUIRE(std::isnan(ptl.gcgvel.s.z));
        REQUIRE(std::isnan(ptl.pos.q1));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticTestParticleVDF.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        std::for_each(begin(particles), std::prev(end(particles)), [&os](auto const &ptl) {
            println(os, "    ", ptl, ", ");
        });
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::MaxwellianVDF::Homogeneous", "[LibPIC::RelativisticVDF::MaxwellianVDF::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 5.35, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1, T2OT1);
    auto const vdf     = RelativisticMaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
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
    auto       particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta1);
        auto const vth2 = vth1 * std::sqrt(T2OT1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticMaxwellianVDF-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::MaxwellianVDF::Inhomogeneous", "[LibPIC::RelativisticVDF::MaxwellianVDF::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1_eq, T2OT1_eq);
    auto const vdf     = RelativisticMaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.0796995979174554 }.epsilon(1e-10));

    std::array const etas{
        0.4287882239954878, 0.49761722933466956, 0.5815167468245395, 0.6800561885442317, 0.788001590917828,
        0.8920754659248894, 0.9704416860618577, 1., 0.9704416860618577, 0.8920754659248894, 0.788001590917828,
        0.6800561885442317, 0.5815167468245395, 0.49761722933466956, 0.4287882239954878, 0.3733660301900688,
        0.32911727304544663, 0.29391455144617107, 0.26596447719479277, 0.24383880547267717, 0.22643378958243887,
        0.21291237930064547, 0.2026501794964986
    };
    std::array const nuv0s{
        FourMFATensor{ 7.6796439042989863566, { 0, 0, 0 }, { 0.27911539238104404737, 0.61594144886886081913, 0.61594144886886081913, 0, 0, 0 } },
        FourMFATensor{ 9.0259795339782940005, { 0, 0, 0 }, { 0.32050341039122298703, 0.81325808234008256647, 0.81325808234008256647, 0, 0, 0 } },
        FourMFATensor{ 10.706184755021030952, { 0, 0, 0 }, { 0.36993938519636432316, 1.0855481415092880226, 1.0855481415092880226, 0, 0, 0 } },
        FourMFATensor{ 12.732781177900619696, { 0, 0, 0 }, { 0.42668819651599032561, 1.4477412024058688989, 1.4477412024058688989, 0, 0, 0 } },
        FourMFATensor{ 15.016692616105265401, { 0, 0, 0 }, { 0.48735966029137123279, 1.8943437716588384934, 1.8943437716588384934, 0, 0, 0 } },
        FourMFATensor{ 17.279918470199586267, { 0, 0, 0 }, { 0.54449711801919620235, 2.3717632904336083399, 2.3717632904336083399, 0, 0, 0 } },
        FourMFATensor{ 19.022668266008288640, { 0, 0, 0 }, { 0.58670448769353888974, 2.7602655816622587714, 2.7602655816622587714, 0, 0, 0 } },
        FourMFATensor{ 19.688441698474679953, { 0, 0, 0 }, { 0.60245142988768141112, 2.9130780612887035019, 2.9130780612887035019, 0, 0, 0 } },
        FourMFATensor{ 19.022668266008288640, { 0, 0, 0 }, { 0.58670448769353888974, 2.7602655816622587714, 2.7602655816622587714, 0, 0, 0 } },
        FourMFATensor{ 17.279918470199586267, { 0, 0, 0 }, { 0.54449711801919620235, 2.3717632904336083399, 2.3717632904336083399, 0, 0, 0 } },
        FourMFATensor{ 15.016692616105265401, { 0, 0, 0 }, { 0.48735966029137123279, 1.8943437716588384934, 1.8943437716588384934, 0, 0, 0 } },
        FourMFATensor{ 12.732781177900619696, { 0, 0, 0 }, { 0.42668819651599032561, 1.4477412024058688989, 1.4477412024058688989, 0, 0, 0 } },
        FourMFATensor{ 10.706184755021030952, { 0, 0, 0 }, { 0.36993938519636432316, 1.0855481415092880226, 1.0855481415092880226, 0, 0, 0 } },
        FourMFATensor{ 9.0259795339782940005, { 0, 0, 0 }, { 0.32050341039122298703, 0.81325808234008256647, 0.81325808234008256647, 0, 0, 0 } },
        FourMFATensor{ 7.6796439042989863566, { 0, 0, 0 }, { 0.27911539238104404737, 0.61594144886886081913, 0.61594144886886081913, 0, 0, 0 } },
        FourMFATensor{ 6.6171230649494150455, { 0, 0, 0 }, { 0.24520396653403486731, 0.47492035609505728333, 0.47492035609505728333, 0, 0, 0 } },
        FourMFATensor{ 5.7829467934930152140, { 0, 0, 0 }, { 0.21773090191556496165, 0.37422770055057413829, 0.37422770055057413829, 0, 0, 0 } },
        FourMFATensor{ 5.1284442913639001205, { 0, 0, 0 }, { 0.19560829100507137746, 0.30192268264068905514, 0.30192268264068905514, 0, 0, 0 } },
        FourMFATensor{ 4.6146462260100493680, { 0, 0, 0 }, { 0.17786822147555930718, 0.24957915752025311429, 0.24957915752025311429, 0, 0, 0 } },
        FourMFATensor{ 4.2116440004148207876, { 0, 0, 0 }, { 0.16371093623219648561, 0.21139778865898520288, 0.21139778865898520288, 0, 0, 0 } },
        FourMFATensor{ 3.8969648050428511432, { 0, 0, 0 }, { 0.15250132374346736519, 0.18342295339037723023, 0.18342295339037723023, 0, 0, 0 } },
        FourMFATensor{ 3.6539349702175152323, { 0, 0, 0 }, { 0.14374758536644874352, 0.16296207621484651296, 0.16296207621484651296, 0, 0, 0 } },
        FourMFATensor{ 3.4703283142123186877, { 0, 0, 0 }, { 0.13707689292660604763, 0.14818475209338716203, 0.14818475209338716203, 0, 0, 0 } },
    };
    static_assert(std::size(etas) == std::size(nuv0s));
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const n0_ref   = etas.at(i);
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = nuv0s.at(i);

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto       particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n0   = *vdf.n0(ptl.pos);
        auto const vth1 = std::sqrt(beta1_eq);
        auto const vth2 = vth1 * std::sqrt(T2OT1_eq * n0);
        auto const uel  = geo.cart_to_mfa(ptl.gcgvel, ptl.pos).s;
        auto const u1   = uel.x;
        auto const u2   = std::sqrt(uel.y * uel.y + uel.z * uel.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticMaxwellianVDF-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::LossconeVDF::BiMax::Homogeneous", "[LibPIC::RelativisticVDF::LossconeVDF::BiMax::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 2.35, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, 0, 2.1);
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1, T2OT1);
    auto const vdf     = RelativisticLossconeVDF(LossconePlasmaDesc{ {}, desc }, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(LossconePlasmaDesc({}, desc)) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
        auto const nuv0_ref = FourTensor{
            17.945115935082544212,
            { 0, 0, 0 },
            { 0.64987333698425020501, 1.4669918147933842523, 1.4669918147933842523, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta1);
        auto const vth2 = vth1 * std::sqrt(T2OT1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-7));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-7));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticLossconeVDF-bi_max-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::LossconeVDF::BiMax::Inhomogeneous", "[LibPIC::RelativisticVDF::LossconeVDF::BiMax::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1_eq, T2OT1_eq);
    auto const vdf     = RelativisticLossconeVDF(LossconePlasmaDesc({}, desc), geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(LossconePlasmaDesc({}, desc)) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.0796995979174554 }.epsilon(1e-7));

    std::array const etas{
        0.42879123633025984, 0.49762030396573637, 0.5815197397970416, 0.6800588645090684, 0.7880036454881502, 0.8920766500089541,
        0.970442038846314, 1., 0.970442038846314, 0.8920766500089541, 0.7880036454881502, 0.6800588645090684, 0.5815197397970416,
        0.49762030396573637, 0.42879123633025984, 0.37336890766975, 0.3291199886157569, 0.29391710380836455, 0.2659668782647602,
        0.24384107315089862, 0.22643594386685753, 0.21291444034995088, 0.20265216678232262
    };
    std::array const eta_bs{
        1.4413912093052352, 1.3022091219893086, 1.1982158749947158, 1.1212616249880305, 1.0659200692069175, 1.0286058654348758,
        1.0070509750175554, 1., 1.0070509750175554, 1.0286058654348758, 1.0659200692069175, 1.1212616249880305, 1.1982158749947158,
        1.3022091219893086, 1.4413912093052352, 1.6281445589143488, 1.881749612870893, 2.2333127829938704, 2.7354240450511313,
        3.482518706776325, 4.657982242968423, 6.657178178048698, 10.466831843938639
    };
    static_assert(std::size(etas) == std::size(eta_bs));
    std::array const nuv0s{
        FourCartTensor{ 7.6796462034444923361, { 0, 0, 0 }, { 0.27911544723131065382, 0.61594196613518792383, 0.61594196613518792383, 0, 0, 0 } },
        FourCartTensor{ 9.0259822949686512317, { 0, 0, 0 }, { 0.32050347314752558603, 0.81325873933827541595, 0.81325873933827541595, 0, 0, 0 } },
        FourCartTensor{ 10.706188106166683482, { 0, 0, 0 }, { 0.36993945747142253921, 1.0855489837493936811, 1.0855489837493936811, 0, 0, 0 } },
        FourCartTensor{ 12.732785257673132406, { 0, 0, 0 }, { 0.42668827983679258331, 1.4477422814519920191, 1.4477422814519920191, 0, 0, 0 } },
        FourCartTensor{ 15.016697535019797982, { 0, 0, 0 }, { 0.48735975557933164382, 1.8943451320783026848, 1.8943451320783026848, 0, 0, 0 } },
        FourCartTensor{ 17.279924236095538959, { 0, 0, 0 }, { 0.54449722473858208627, 2.3717649422882374211, 2.3717649422882374211, 0, 0, 0 } },
        FourCartTensor{ 19.022674692923139617, { 0, 0, 0 }, { 0.58670460296282556101, 2.7602674652333742955, 2.7602674652333742955, 0, 0, 0 } },
        FourCartTensor{ 19.688448379713204162, { 0, 0, 0 }, { 0.60245154837038406015, 2.9130800348860912408, 2.9130800348860912408, 0, 0, 0 } },
        FourCartTensor{ 19.022674692923139617, { 0, 0, 0 }, { 0.58670460296282556101, 2.7602674652333742955, 2.7602674652333742955, 0, 0, 0 } },
        FourCartTensor{ 17.279924236095538959, { 0, 0, 0 }, { 0.54449722473858208627, 2.3717649422882374211, 2.3717649422882374211, 0, 0, 0 } },
        FourCartTensor{ 15.016697535019797982, { 0, 0, 0 }, { 0.48735975557933164382, 1.8943451320783026848, 1.8943451320783026848, 0, 0, 0 } },
        FourCartTensor{ 12.732785257673132406, { 0, 0, 0 }, { 0.42668827983679258331, 1.4477422814519920191, 1.4477422814519920191, 0, 0, 0 } },
        FourCartTensor{ 10.706188106166683482, { 0, 0, 0 }, { 0.36993945747142253921, 1.0855489837493936811, 1.0855489837493936811, 0, 0, 0 } },
        FourCartTensor{ 9.0259822949686512317, { 0, 0, 0 }, { 0.32050347314752558603, 0.81325873933827541595, 0.81325873933827541595, 0, 0, 0 } },
        FourCartTensor{ 7.6796462034444923361, { 0, 0, 0 }, { 0.27911544723131065382, 0.61594196613518792383, 0.61594196613518792383, 0, 0, 0 } },
        FourCartTensor{ 6.6171250078839518594, { 0, 0, 0 }, { 0.24520401494134322351, 0.47492076954370293640, 0.47492076954370293640, 0, 0, 0 } },
        FourCartTensor{ 5.7829484627882870029, { 0, 0, 0 }, { 0.21773094512193727490, 0.37422803712739766135, 0.37422803712739766135, 0, 0, 0 } },
        FourCartTensor{ 5.1284457503005871359, { 0, 0, 0 }, { 0.19560833003546504449, 0.30192296215863922981, 0.30192296215863922981, 0, 0, 0 } },
        FourCartTensor{ 4.6146475229967451881, { 0, 0, 0 }, { 0.17786825716972731737, 0.24957939449787741593, 0.24957939449787741593, 0, 0, 0 } },
        FourCartTensor{ 4.2116451728637711582, { 0, 0, 0 }, { 0.16371096928343245591, 0.21139799380882140500, 0.21139799380882140500, 0, 0, 0 } },
        FourCartTensor{ 3.8969658825016080250, { 0, 0, 0 }, { 0.15250135473769352301, 0.18342313473206131591, 0.18342313473206131591, 0, 0, 0 } },
        FourCartTensor{ 3.6539359769711730053, { 0, 0, 0 }, { 0.14374761482443654259, 0.16296223988764688140, 0.16296223988764688140, 0, 0, 0 } },
        FourCartTensor{ 3.4703292719587448545, { 0, 0, 0 }, { 0.13707692136679963668, 0.14818490296684946883, 0.14818490296684946883, 0, 0, 0 } },
    };
    static_assert(std::size(etas) == std::size(nuv0s));
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta     = etas.at(i);
        auto const eta_b   = eta_bs.at(i);
        auto const beta_eq = 1e-10;

        auto const n0_ref   = (eta - beta_eq * eta_b) / (1 - beta_eq);
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = nuv0s.at(i);

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n0   = *vdf.n0(ptl.pos);
        auto const vth1 = std::sqrt(beta1_eq);
        auto const vth2 = vth1 * std::sqrt(T2OT1_eq * n0);
        auto const uel  = geo.cart_to_mfa(ptl.gcgvel, ptl.pos).s;
        auto const u1   = uel.x;
        auto const u2   = std::sqrt(uel.y * uel.y + uel.z * uel.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-7));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-7));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticLossconeVDF-bi_max-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::LossconeVDF::Homogeneous", "[LibPIC::RelativisticVDF::LossconeVDF::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35, beta_eq = .9;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0, 2.1);
    auto const desc    = LossconePlasmaDesc({ beta_eq }, kinetic, beta1_eq, T2OT1_eq / (1 + beta_eq));
    auto const vdf     = RelativisticLossconeVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = RelativisticLossconeVDF(LossconePlasmaDesc({ beta_eq }, g_desc), geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const B_div_B0          = geo.Bmag_div_B0(ptl.pos);
        auto const vth_ratio_squared = T2OT1_eq / (1 + beta_eq);
        auto const eta               = std::pow((1 - 1 / B_div_B0) * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const eta_b             = std::pow((1 - 1 / B_div_B0) * beta_eq * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const beta              = beta_eq * eta_b / eta;
        auto const vth1              = std::sqrt(beta1_eq);
        auto const vth2              = vth1 * std::sqrt(vth_ratio_squared * eta);
        auto const u_co              = geo.cart_to_mfa(ptl.gcgvel, ptl.pos).s;
        auto const u1                = u_co.x;
        auto const u2                = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = eta * (1 - beta) / (1 - beta_eq)
            * std::exp(-u1 * u1 / (vth1 * vth1))
            * (std::exp(-u2 * u2 / (vth2 * vth2)) - std::exp(-u2 * u2 / (beta * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2) / (1 - beta);
        auto const g_ref
            = (eta - beta_eq * eta_b) / (1 - beta_eq)
            * std::exp(-u1 * u1 / (desc.marker_temp_ratio * vth1 * vth1))
            * (std::exp(-u2 * u2 / (desc.marker_temp_ratio * vth2 * vth2)) - std::exp(-u2 * u2 / (beta * desc.marker_temp_ratio * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio)) / (1 - beta);

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticLossconeVDF-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::LossconeVDF::Inhomogeneous", "[LibPIC::RelativisticVDF::LossconeVDF::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35, beta_eq = .9;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = LossconePlasmaDesc({ beta_eq }, kinetic, beta1_eq, T2OT1_eq / (1 + beta_eq));
    auto const vdf     = RelativisticLossconeVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = RelativisticLossconeVDF(LossconePlasmaDesc({ beta_eq }, g_desc), geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.09452655524436228 }.epsilon(1e-10));

    std::array const etas{
        0.64264404305085, 0.7035216274440845, 0.7689973601535578, 0.835852333403872, 0.8990376137203975, 0.9519272903927195,
        0.9874454952939664, 1., 0.9874454952939664, 0.9519272903927195, 0.8990376137203975, 0.835852333403872, 0.7689973601535578,
        0.7035216274440845, 0.64264404305085, 0.5880359901402682, 0.5402813355647949, 0.4993020809205341, 0.46467416286831936,
        0.4358328683746492, 0.4121939193295821, 0.39321849079530274, 0.37844426744095166
    };
    std::array const eta_bs{
        0.6803461521487998, 0.7374252176467319, 0.7975679579815613, 0.8576845064678151, 0.9133372658253722, 0.9590769728307393,
        0.9893716614043512, 1., 0.9893716614043512, 0.9590769728307393, 0.9133372658253722, 0.8576845064678151, 0.7975679579815613,
        0.7374252176467319, 0.6803461521487998, 0.6281659134692374, 0.5817544384045442, 0.5413336389508113, 0.5067409023782806,
        0.47761814536264413, 0.45353476372795026, 0.4340615580713739, 0.41881195104513863
    };
    static_assert(std::size(etas) == std::size(eta_bs));
    std::array const nuv0s{
        FourCartTensor{ 5.6768858251156881778, { 0, 0, 0 }, { 0.18952147890249734785, 0.65505382322221805680, 0.65505382322221805680, 0, 0, 0 } },
        FourCartTensor{ 7.5325158508074165908, { 0, 0, 0 }, { 0.24671496338192841491, 0.92534625052598151740, 0.92534625052598151740, 0, 0, 0 } },
        FourCartTensor{ 9.7812977322508132261, { 0, 0, 0 }, { 0.31407303922865326129, 1.2758926848675709032, 1.2758926848675709032, 0, 0, 0 } },
        FourCartTensor{ 12.346537970738083345, { 0, 0, 0 }, { 0.38875302108018805480, 1.7011247602191639228, 1.7011247602191639228, 0, 0, 0 } },
        FourCartTensor{ 15.019336503132997507, { 0, 0, 0 }, { 0.46451380274049147712, 2.1682593988373635163, 2.1682593988373635163, 0, 0, 0 } },
        FourCartTensor{ 17.441001153629063225, { 0, 0, 0 }, { 0.53160336653264217421, 2.6097130304620668184, 2.6097130304620668184, 0, 0, 0 } },
        FourCartTensor{ 19.161046152724402702, { 0, 0, 0 }, { 0.57845799910457196269, 2.9326088394623877065, 2.9326088394623877065, 0, 0, 0 } },
        FourCartTensor{ 19.786975759925791607, { 0, 0, 0 }, { 0.59535600305974989421, 3.0518980566112574593, 3.0518980566112574593, 0, 0, 0 } },
        FourCartTensor{ 19.161046152724402702, { 0, 0, 0 }, { 0.57845799910457196269, 2.9326088394623877065, 2.9326088394623877065, 0, 0, 0 } },
        FourCartTensor{ 17.441001153629063225, { 0, 0, 0 }, { 0.53160336653264217421, 2.6097130304620668184, 2.6097130304620668184, 0, 0, 0 } },
        FourCartTensor{ 15.019336503132997507, { 0, 0, 0 }, { 0.46451380274049147712, 2.1682593988373635163, 2.1682593988373635163, 0, 0, 0 } },
        FourCartTensor{ 12.346537970738083345, { 0, 0, 0 }, { 0.38875302108018805480, 1.7011247602191639228, 1.7011247602191639228, 0, 0, 0 } },
        FourCartTensor{ 9.7812977322508132261, { 0, 0, 0 }, { 0.31407303922865326129, 1.2758926848675709032, 1.2758926848675709032, 0, 0, 0 } },
        FourCartTensor{ 7.5325158508074165908, { 0, 0, 0 }, { 0.24671496338192841491, 0.92534625052598151740, 0.92534625052598151740, 0, 0, 0 } },
        FourCartTensor{ 5.6768858251156881778, { 0, 0, 0 }, { 0.18952147890249734785, 0.65505382322221805680, 0.65505382322221805680, 0, 0, 0 } },
        FourCartTensor{ 4.2060393451634512374, { 0, 0, 0 }, { 0.14292341062034225052, 0.45570969262913385567, 0.45570969262913385567, 0, 0, 0 } },
        FourCartTensor{ 3.0703815831021477045, { 0, 0, 0 }, { 0.10600849718852121961, 0.31284019239210780761, 0.31284019239210780761, 0, 0, 0 } },
        FourCartTensor{ 2.2081068292425873878, { 0, 0, 0 }, { 0.077314891465893501032, 0.21222227141461277866, 0.21222227141461277866, 0, 0, 0 } },
        FourCartTensor{ 1.5604013923274642206, { 0, 0, 0 }, { 0.055303070250483712944, 0.14206474227103205177, 0.14206474227103205177, 0, 0, 0 } },
        FourCartTensor{ 1.0775500443065417766, { 0, 0, 0 }, { 0.038585760921017189795, 0.093410558804140758626, 0.093410558804140758626, 0, 0, 0 } },
        FourCartTensor{ 0.72017173291310576655, { 0, 0, 0 }, { 0.026010966275987520979, 0.059789530693340926792, 0.059789530693340926792, 0, 0, 0 } },
        FourCartTensor{ 0.45830987340083789716, { 0, 0, 0 }, { 0.016669241832377050538, 0.036670432632644384130, 0.036670432632644384130, 0, 0, 0 } },
        FourCartTensor{ 0.26984458231288988017, { 0, 0, 0 }, { 0.0098687138099334756336, 0.020947436381219895762, 0.020947436381219895762, 0, 0, 0 } },
    };
    static_assert(std::size(etas) == std::size(nuv0s));
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta   = etas.at(i);
        auto const eta_b = eta_bs.at(i);

        auto const n0_ref   = (eta - beta_eq * eta_b) / (1 - beta_eq);
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = nuv0s.at(i);

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta1) }, CartTensor{ desc.beta1 } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta1),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta1
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(2e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(2e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(2e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const B_div_B0          = geo.Bmag_div_B0(ptl.pos);
        auto const vth_ratio_squared = T2OT1_eq / (1 + beta_eq);
        auto const eta               = std::pow((1 - 1 / B_div_B0) * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const eta_b             = std::pow((1 - 1 / B_div_B0) * beta_eq * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const beta              = beta_eq * eta_b / eta;
        auto const vth1              = std::sqrt(beta1_eq);
        auto const vth2              = vth1 * std::sqrt(vth_ratio_squared * eta);
        auto const uel               = geo.cart_to_mfa(ptl.gcgvel, ptl.pos).s;
        auto const u1                = uel.x;
        auto const u2                = std::sqrt(uel.y * uel.y + uel.z * uel.z);
        auto const f_ref
            = eta * (1 - beta) / (1 - beta_eq)
            * std::exp(-u1 * u1 / (vth1 * vth1))
            * (std::exp(-u2 * u2 / (vth2 * vth2)) - std::exp(-u2 * u2 / (beta * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2) / (1 - beta);
        auto const g_ref
            = (eta - beta_eq * eta_b) / (1 - beta_eq)
            * std::exp(-u1 * u1 / (desc.marker_temp_ratio * vth1 * vth1))
            * (std::exp(-u2 * u2 / (desc.marker_temp_ratio * vth2 * vth2)) - std::exp(-u2 * u2 / (beta * desc.marker_temp_ratio * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio)) / (1 - beta);

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticLossconeVDF-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::Maxwellian::Homogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::Maxwellian::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 0, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, 0, 2.1);
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc(kinetic, beta, 1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        REQUIRE(vdf.n0(pos) == g_vdf.n0(pos));
        REQUIRE(vdf.nV0(pos) == g_vdf.nV0(pos));
        REQUIRE(vdf.nuv0(pos) == g_vdf.nuv0(pos));
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * g_vdf.f0(ptl) / g_vdf.g0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.g0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ g_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta);
        auto const vth2 = vth1 * std::sqrt(g_desc.T2_T1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-maxwellian-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::Maxwellian::Inhomogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::Maxwellian::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 0, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc(kinetic, beta, 1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        REQUIRE(vdf.n0(pos) == g_vdf.n0(pos));
        REQUIRE(vdf.nV0(pos) == g_vdf.nV0(pos));
        REQUIRE(vdf.nuv0(pos) == g_vdf.nuv0(pos));
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ g_vdf.f0(ptl) / g_vdf.g0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.g0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ g_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta);
        auto const vth2 = vth1 * std::sqrt(g_desc.T2_T1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-maxwellian-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::IsotropicShell::Homogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::IsotropicShell::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 10, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, 0, 2.1);
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc(kinetic, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = RelativisticPartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
        auto const nuv0_ref = FourTensor{
            43.653323887577492712,
            { 0, 0, 0 },
            { 12.585699767186241615, 12.585696243854586740, 12.585696243854586740, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth  = std::sqrt(beta);
        auto const b    = vs / vth;
        auto const Ab   = .5 * (b * std::exp(-b * b) + 2 / M_2_SQRTPI * (.5 + b * b) * std::erfc(-b));
        auto const Bz   = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u    = std::sqrt(dot(u_co, u_co));
        auto const cosa = u_co.x / u;
        auto const f_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * Ab * Bz * vth * vth * vth);

        auto const marker_b  = vs / (vth * std::sqrt(desc.marker_temp_ratio));
        auto const marker_Ab = .5 * (marker_b * std::exp(-marker_b * marker_b) + 2 / M_2_SQRTPI * (.5 + marker_b * marker_b) * std::erfc(-marker_b));
        auto const g_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth * desc.marker_temp_ratio)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * marker_Ab * Bz * vth * vth * vth * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-isotropic_shell-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::IsotropicShell::Inhomogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::IsotropicShell::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 10, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc(kinetic, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = RelativisticPartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
        auto const nuv0_ref = FourTensor{
            43.653323887577492712,
            { 0, 0, 0 },
            { 12.585699767186241615, 12.585696243854586740, 12.585696243854586740, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth  = std::sqrt(beta);
        auto const b    = vs / vth;
        auto const Ab   = .5 * (b * std::exp(-b * b) + 2 / M_2_SQRTPI * (.5 + b * b) * std::erfc(-b));
        auto const Bz   = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u    = std::sqrt(dot(u_co, u_co));
        auto const cosa = u_co.x / u;
        auto const f_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * Ab * Bz * vth * vth * vth);

        auto const marker_b  = vs / (vth * std::sqrt(desc.marker_temp_ratio));
        auto const marker_Ab = .5 * (marker_b * std::exp(-marker_b * marker_b) + 2 / M_2_SQRTPI * (.5 + marker_b * marker_b) * std::erfc(-marker_b));
        auto const g_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth * desc.marker_temp_ratio)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * marker_Ab * Bz * vth * vth * vth * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-isotropic_shell-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::AnisotropicShell::Homogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::AnisotropicShell::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 30;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = RelativisticPartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-anisotropic_shell-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::PartialShellVDF::AnisotropicShell::Inhomogeneous", "[LibPIC::RelativisticVDF::PartialShellVDF::AnisotropicShell::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 2;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 10;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = RelativisticPartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = RelativisticPartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.1108013193858871 }.epsilon(1e-10));

    std::array const etas{
        0.16070869470151475, 0.26703071776468223, 0.40485697458795117, 0.5642234318909161, 0.7267234017099303,
        0.868463747345441, 0.9654769492087792, 1., 0.9654769492087792, 0.868463747345441, 0.7267234017099303,
        0.5642234318909161, 0.40485697458795117, 0.26703071776468223, 0.16070869470151475, 0.08739009484781933,
        0.0423729187357854, 0.017993222556708512, 0.0065264207733627755, 0.001950941625094117, 0.00045560222393496486,
        0.00007636454203949487, 7.940057378694681e-6
    };
    std::array const nuv0s{
        FourMFATensor{ 3.1055670106293877808, { 0, 0, 0 }, { 0.073928194638746558276, 0.44356915646249300833, 0.44356915646249300833, 0, 0, 0 } },
        FourMFATensor{ 5.1601550834254119593, { 0, 0, 0 }, { 0.12283777747119964396, 0.73702664593502986712, 0.73702664593502986712, 0, 0, 0 } },
        FourMFATensor{ 7.8235372805358895931, { 0, 0, 0 }, { 0.18623973814100064361, 1.1174384002027601959, 1.1174384002027601959, 0, 0, 0 } },
        FourMFATensor{ 10.903166626789921878, { 0, 0, 0 }, { 0.25955048524316109981, 1.5573028715406984634, 1.5573028715406984634, 0, 0, 0 } },
        FourMFATensor{ 14.043348596629822822, { 0, 0, 0 }, { 0.33430269090249381536, 2.0058160939999774008, 2.0058160939999774008, 0, 0, 0 } },
        FourMFATensor{ 16.782367430043940715, { 0, 0, 0 }, { 0.39950518588739331038, 2.3970310538813803802, 2.3970310538813803802, 0, 0, 0 } },
        FourMFATensor{ 18.657069977170490205, { 0, 0, 0 }, { 0.44413258382128534274, 2.6647954346211411014, 2.6647954346211411014, 0, 0, 0 } },
        FourMFATensor{ 19.324200326543479633, { 0, 0, 0 }, { 0.46001365872614341512, 2.7600818816078156459, 2.7600818816078156459, 0, 0, 0 } },
        FourMFATensor{ 18.657069977170490205, { 0, 0, 0 }, { 0.44413258382128534274, 2.6647954346211411014, 2.6647954346211411014, 0, 0, 0 } },
        FourMFATensor{ 16.782367430043940715, { 0, 0, 0 }, { 0.39950518588739331038, 2.3970310538813803802, 2.3970310538813803802, 0, 0, 0 } },
        FourMFATensor{ 14.043348596629822822, { 0, 0, 0 }, { 0.33430269090249381536, 2.0058160939999774008, 2.0058160939999774008, 0, 0, 0 } },
        FourMFATensor{ 10.903166626789921878, { 0, 0, 0 }, { 0.25955048524316109981, 1.5573028715406984634, 1.5573028715406984634, 0, 0, 0 } },
        FourMFATensor{ 7.8235372805358895931, { 0, 0, 0 }, { 0.18623973814100064361, 1.1174384002027601959, 1.1174384002027601959, 0, 0, 0 } },
        FourMFATensor{ 5.1601550834254119593, { 0, 0, 0 }, { 0.12283777747119964396, 0.73702664593502986712, 0.73702664593502986712, 0, 0, 0 } },
        FourMFATensor{ 3.1055670106293877808, { 0, 0, 0 }, { 0.073928194638746558276, 0.44356915646249300833, 0.44356915646249300833, 0, 0, 0 } },
        FourMFATensor{ 1.6887436993948958808, { 0, 0, 0 }, { 0.040200637267370047112, 0.24120381742145469151, 0.24120381742145469151, 0, 0, 0 } },
        FourMFATensor{ 0.81882277007066461838, { 0, 0, 0 }, { 0.019492121378554189137, 0.11695272527348161973, 0.11695272527348161973, 0, 0, 0 } },
        FourMFATensor{ 0.34770463720591615608, { 0, 0, 0 }, { 0.0082771281405852551588, 0.049662767570508248305, 0.049662767570508248305, 0, 0, 0 } },
        FourMFATensor{ 0.12611786243977704536, { 0, 0, 0 }, { 0.0030022426983409167961, 0.018013455728307467552, 0.018013455728307467552, 0, 0, 0 } },
        FourMFATensor{ 0.037700386788710994745, { 0, 0, 0 }, { 0.00089745979492067255463, 0.0053847586314967801208, 0.0053847586314967801208, 0, 0, 0 } },
        FourMFATensor{ 0.0088041486445379832371, { 0, 0, 0 }, { 0.00020958324595609078298, 0.0012574994435031233622, 0.0012574994435031233622, 0, 0, 0 } },
        FourMFATensor{ 0.0014756837082159494131, { 0, 0, 0 }, { 0.000035128732380534419104, 0.00021077238888048821638, 0.00021077238888048821638, 0, 0, 0 } },
        FourMFATensor{ 0.00015343525939014575300, { 0, 0, 0 }, { 3.6525348452688510663e-6, 0.000021915208509861638437, 0.000021915208509861638437, 0, 0, 0 } },
    };
    static_assert(std::size(etas) == std::size(nuv0s));
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const n0_ref   = etas.at(i);
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = nuv0s.at(i);

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticPartialShellVDF-anisotropic_shell-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::RejectionSampling", "[LibPIC::RelativisticVDF::CounterBeamVDF::RejectionSampling]")
{
    long const n_samples = 500000;
    Real const nu        = 1;

    std::vector<Real> samples(n_samples);
    std::generate_n(begin(samples), samples.size(), [nu] {
        return RelativisticCounterBeamVDF::RejectionSampling{ nu }.sample();
    });

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-RejectionSampling.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{', nu, ", ");
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < samples.size() - 1; ++i) {
            println(os, "    ", samples[i], ", ");
        }
        println(os, "    ", samples.back());
        println(os, '}');
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::Maxwellian::Homogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::Maxwellian::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 100000, vs = 0, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, 0, 2.1);
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc(kinetic, beta, 1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        REQUIRE(vdf.n0(pos) == g_vdf.n0(pos));
        REQUIRE(vdf.nV0(pos) == g_vdf.nV0(pos));
        REQUIRE(vdf.nuv0(pos) == g_vdf.nuv0(pos));
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * g_vdf.f0(ptl) / g_vdf.g0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.g0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ g_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta);
        auto const vth2 = vth1 * std::sqrt(g_desc.T2_T1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-maxwellian-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::Maxwellian::Inhomogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::Maxwellian::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 100'000, vs = 0, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc(kinetic, beta, 1);
    auto const g_vdf  = RelativisticMaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        REQUIRE(vdf.n0(pos) == g_vdf.n0(pos));
        REQUIRE(vdf.nV0(pos) == g_vdf.nV0(pos));
        REQUIRE(vdf.nuv0(pos) == g_vdf.nuv0(pos));
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(1e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(1e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(1e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ g_vdf.f0(ptl) / g_vdf.g0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.g0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ g_vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth1 = std::sqrt(beta);
        auto const vth2 = vth1 * std::sqrt(g_desc.T2_T1);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u1   = u_co.x;
        auto const u2   = std::sqrt(u_co.y * u_co.y + u_co.z * u_co.z);
        auto const f_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1) - u2 * u2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n0 * std::exp(-u1 * u1 / (vth1 * vth1 * desc.marker_temp_ratio) - u2 * u2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-maxwellian-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::IsotropicShell::Homogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::IsotropicShell::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 100'000, vs = 10, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, 0, 2.1);
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc(kinetic, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = RelativisticCounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
        auto const nuv0_ref = FourTensor{
            43.653323887577492712,
            { 0, 0, 0 },
            { 12.585699767186241615, 12.585696243854586740, 12.585696243854586740, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(2e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(4e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const zeta = 0;
        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth  = std::sqrt(beta);
        auto const b    = vs / vth;
        auto const Ab   = .5 * (b * std::exp(-b * b) + 2 / M_2_SQRTPI * (.5 + b * b) * std::erfc(-b));
        auto const Bz   = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u    = std::sqrt(dot(u_co, u_co));
        auto const cosa = u_co.x / u;
        auto const f_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * Ab * Bz * vth * vth * vth);

        auto const marker_b  = vs / (vth * std::sqrt(desc.marker_temp_ratio));
        auto const marker_Ab = .5 * (marker_b * std::exp(-marker_b * marker_b) + 2 / M_2_SQRTPI * (.5 + marker_b * marker_b) * std::erfc(-marker_b));
        auto const g_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth * desc.marker_temp_ratio)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * marker_Ab * Bz * vth * vth * vth * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-isotropic_shell-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::IsotropicShell::Inhomogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::IsotropicShell::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 100'000, vs = 10, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc(kinetic, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = RelativisticCounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = n0_ref * Vector{ Vd, 0, 0 };
        auto const nuv0_ref = FourTensor{
            43.653323887577492712,
            { 0, 0, 0 },
            { 12.585699767186241615, 12.585696243854586740, 12.585696243854586740, 0, 0, 0 },
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(1e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(2e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(4e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const zeta = 0;
        auto const gd   = c / std::sqrt((c - Vd) * (c + Vd));
        auto const n0   = *vdf.n0(ptl.pos) / gd;
        auto const vth  = std::sqrt(beta);
        auto const b    = vs / vth;
        auto const Ab   = .5 * (b * std::exp(-b * b) + 2 / M_2_SQRTPI * (.5 + b * b) * std::erfc(-b));
        auto const Bz   = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
        auto const u_co = lorentz_boost<+1>(geo.cart_to_mfa(ptl.gcgvel, ptl.pos), Vd / c, gd).s;
        auto const u    = std::sqrt(dot(u_co, u_co));
        auto const cosa = u_co.x / u;
        auto const f_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * Ab * Bz * vth * vth * vth);

        auto const marker_b  = vs / (vth * std::sqrt(desc.marker_temp_ratio));
        auto const marker_Ab = .5 * (marker_b * std::exp(-marker_b * marker_b) + 2 / M_2_SQRTPI * (.5 + marker_b * marker_b) * std::erfc(-marker_b));
        auto const g_ref
            = n0 * std::exp(-std::pow(u - vs, 2) / (vth * vth * desc.marker_temp_ratio)) * std::pow((1 - cosa) * (1 + cosa), .5 * zeta)
            / (2 * M_PI * marker_Ab * Bz * vth * vth * vth * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-isotropic_shell-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::CounterBeam::Homogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::CounterBeam::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.1, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = RelativisticCounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(1e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(2e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(3e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n  = *vdf.n0(ptl.pos);
        auto const nu = std::sqrt(geo.Bmag_div_B0(ptl.pos)) * nu0;
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-counter_beam-homogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}

TEST_CASE("Test LibPIC::RelativisticVDF::CounterBeamVDF::CounterBeam::Inhomogeneous", "[LibPIC::RelativisticVDF::CounterBeamVDF::CounterBeam::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.2, vs = 2;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, 0.001, 2.1);
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = RelativisticCounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = RelativisticCounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.02072000799696996 }.epsilon(1e-8));

    std::array const etas{
        1.4561623198207398477, 1.3112428726471929696, 1.2036232244145275150,
        1.1243388313063897854, 1.0675036398265502768, 1.0292671327256639469,
        1.0072102956472859248, 1.0000000000000000000, 1.0072102956472859248,
        1.0292671327256639469, 1.0675036398265502768, 1.1243388313063897854,
        1.2036232244145275150, 1.3112428726471929696, 1.4561623198207398477,
        1.6522706092531636024, 1.9218119442765297933, 2.3022398335834806105,
        2.8613029050290332833, 3.7337267517695922336, 5.2131172452256597794,
        7.9484257778150615437, 13.162564632484334837
    };
    std::array const nuv0s{
        FourMFATensor{ 28.139262088250962535, { 0, 0, 0 }, { 8.1888430317870835040, 0.25964253144427035247, 0.25964253144427035247, 0, 0, 0 } },
        FourMFATensor{ 25.338800909955651264, { 0, 0, 0 }, { 7.4206369111517824777, 0.21042263533751637272, 0.21042263533751637272, 0, 0, 0 } },
        FourMFATensor{ 23.259130736673178319, { 0, 0, 0 }, { 6.8434167242137364795, 0.17723972340593369124, 0.17723972340593369124, 0, 0, 0 } },
        FourMFATensor{ 21.727018411758113103, { 0, 0, 0 }, { 6.4145111840314630314, 0.15462460061590588278, 0.15462460061590588278, 0, 0, 0 } },
        FourMFATensor{ 20.628720314886599851, { 0, 0, 0 }, { 6.1051410647187918102, 0.13936701976193299757, 0.13936701976193299757, 0, 0, 0 } },
        FourMFATensor{ 19.889828052577730233, { 0, 0, 0 }, { 5.8961132971336986941, 0.12955010216944964907, 0.12955010216944964907, 0, 0, 0 } },
        FourMFATensor{ 19.463595963412984702, { 0, 0, 0 }, { 5.7752076480832963412, 0.12405088107069810188, 0.12405088107069810188, 0, 0, 0 } },
        FourMFATensor{ 19.324262324081288966, { 0, 0, 0 }, { 5.7356321246716399642, 0.12227917140125904583, 0.12227917140125904583, 0, 0, 0 } },
        FourMFATensor{ 19.463595963412984702, { 0, 0, 0 }, { 5.7752076480832963412, 0.12405088107069810188, 0.12405088107069810188, 0, 0, 0 } },
        FourMFATensor{ 19.889828052577730233, { 0, 0, 0 }, { 5.8961132971336986941, 0.12955010216944964907, 0.12955010216944964907, 0, 0, 0 } },
        FourMFATensor{ 20.628720314886599851, { 0, 0, 0 }, { 6.1051410647187918102, 0.13936701976193299757, 0.13936701976193299757, 0, 0, 0 } },
        FourMFATensor{ 21.727018411758113103, { 0, 0, 0 }, { 6.4145111840314630314, 0.15462460061590588278, 0.15462460061590588278, 0, 0, 0 } },
        FourMFATensor{ 23.259130736673178319, { 0, 0, 0 }, { 6.8434167242137364795, 0.17723972340593369124, 0.17723972340593369124, 0, 0, 0 } },
        FourMFATensor{ 25.338800909955651264, { 0, 0, 0 }, { 7.4206369111517824777, 0.21042263533751637272, 0.21042263533751637272, 0, 0, 0 } },
        FourMFATensor{ 28.139262088250962535, { 0, 0, 0 }, { 8.1888430317870835040, 0.25964253144427035247, 0.25964253144427035247, 0, 0, 0 } },
        FourMFATensor{ 31.928905565789317933, { 0, 0, 0 }, { 9.2117294141707581900, 0.33458135012858708501, 0.33458135012858708501, 0, 0, 0 } },
        FourMFATensor{ 37.137576292516556009, { 0, 0, 0 }, { 10.586052446889082645, 0.45337232521813053410, 0.45337232521813053410, 0, 0, 0 } },
        FourMFATensor{ 44.489041022887121812, { 0, 0, 0 }, { 12.462430337960597626, 0.65269651069099465790, 0.65269651069099465790, 0, 0, 0 } },
        FourMFATensor{ 55.292487830698668461, { 0, 0, 0 }, { 15.081815508219525768, 1.0146521098282532680, 1.0146521098282532680, 0, 0, 0 } },
        FourMFATensor{ 72.151381099966030774, { 0, 0, 0 }, { 18.838267275385792487, 1.7450509508974090256, 1.7450509508974090256, 0, 0, 0 } },
        FourMFATensor{ 100.73941956943916409, { 0, 0, 0 }, { 24.374027964013770031, 3.4006794957028332327, 3.4006794957028332327, 0, 0, 0 } },
        FourMFATensor{ 153.59707005732403218, { 0, 0, 0 }, { 32.660454026940321626, 7.4362817888382357623, 7.4362817888382357623, 0, 0, 0 } },
        FourMFATensor{ 254.35613464535347816, { 0, 0, 0 }, { 44.830535452879175295, 16.941979097241720353, 16.941979097241720353, 0, 0, 0 } },
    };
    static_assert(std::size(etas) == std::size(nuv0s));
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const n0_ref   = etas.at(i);
        auto const nV0_ref  = Vector{};
        auto const nuv0_ref = nuv0s.at(i);

        INFO("i = " << i << ", q1 = " << q1);
        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nuv0(pos), pos) == nuv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nuv0 = FourCartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nuv0 += vdf.nuv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nuv0 /= *n0 * FourCartTensor{ c * c, CartVector{ c * std::sqrt(desc.beta) }, CartTensor{ desc.beta } };
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.velocity(c) / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), FourCartTensor{}, [&](FourCartTensor const &sum, Particle const &ptl) {
                auto const v    = ptl.velocity(c);
                auto const gcgv = ptl.gcgvel;
                auto const u    = gcgv.s;
                auto const ft   = FourCartTensor{
                    c * gcgv.t / (c * c),
                    c * gcgv.s / c / std::sqrt(desc.beta),
                    CartTensor{ v.x * u.x, v.y * u.y, v.z * u.z, v.x * u.y, v.y * u.z, v.z * u.x } / desc.beta
                };
                return sum + ft * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(*stress_energy.tt == Approx{ *nuv0.tt }.epsilon(1e-2));
        CHECK(stress_energy.ts.x == Approx{ nuv0.ts.x }.margin(2e-2));
        CHECK(stress_energy.ts.y == Approx{ nuv0.ts.y }.margin(2e-2));
        CHECK(stress_energy.ts.z == Approx{ nuv0.ts.z }.margin(2e-2));
        CHECK(stress_energy.ss.xx == Approx{ nuv0.ss.xx }.epsilon(1e-2));
        CHECK(stress_energy.ss.yy == Approx{ nuv0.ss.yy }.epsilon(4e-2));
        CHECK(stress_energy.ss.zz == Approx{ nuv0.ss.zz }.epsilon(4e-2));
        CHECK(stress_energy.ss.xy == Approx{ nuv0.ss.xy }.margin(1e-2));
        CHECK(stress_energy.ss.yz == Approx{ nuv0.ss.yz }.margin(1e-2));
        CHECK(stress_energy.ss.zx == Approx{ nuv0.ss.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-15));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-10));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.gcgvel.t == Approx{ std::sqrt(c * c + dot(ptl.gcgvel.s, ptl.gcgvel.s)) }.epsilon(1e-10));

        auto const n  = *vdf.n0(ptl.pos);
        auto const nu = std::sqrt(geo.Bmag_div_B0(ptl.pos)) * nu0;
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const g_vel = ptl.gcgvel.s;
            auto const v     = std::sqrt(dot(g_vel, g_vel)) - desc.vs;
            auto const alpha = std::acos(g_vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/RelativisticCounterBeamVDF-counter_beam-inhomogeneous.m" };
        os.setf(os.fixed);
        os.precision(20);
        static_assert(n_samples > 0);
        println(os, '{');
        for (unsigned long i = 0; i < particles.size() - 1; ++i) {
            println(os, "    ", particles[i], ", ");
        }
        println(os, "    ", particles.back());
        println(os, '}');
        os.close();
    }
}
