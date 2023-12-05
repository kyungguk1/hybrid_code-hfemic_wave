/*
 * Copyright (c) 2021-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Misc/Faddeeva.hh>
#include <PIC/UTL/println.h>
#include <PIC/VDF/CounterBeamVDF.h>
#include <PIC/VDF/LossconeVDF.h>
#include <PIC/VDF/MaxwellianVDF.h>
#include <PIC/VDF/PartialShellVDF.h>
#include <PIC/VDF/TestParticleVDF.h>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace {
constexpr bool dump_samples         = false;
constexpr bool enable_moment_checks = false;

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

TEST_CASE("Test LibPIC::VDF::TestParticleVDF", "[LibPIC::VDF::TestParticleVDF]")
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
    auto const vdf = TestParticleVDF(desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    CHECK(vdf.initial_number_of_test_particles == Nptls);
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
        REQUIRE(vdf.g0(ptl) == 1);
    }
    {
        auto const &ptl = particles.back();

        REQUIRE(std::isnan(ptl.vel.x));
        REQUIRE(std::isnan(ptl.vel.y));
        REQUIRE(std::isnan(ptl.vel.z));
        REQUIRE(std::isnan(ptl.pos.q1));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/TestParticleVDF.m" };
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

TEST_CASE("Test LibPIC::VDF::MaxwellianVDF::Homogeneous", "[LibPIC::VDF::MaxwellianVDF::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = .1, T2OT1 = 5.35, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1, T2OT1);
    auto const vdf     = MaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 + n0_ref * Vd * Vd,
            desc.beta1 / 2 * desc.T2_T1,
            desc.beta1 / 2 * desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta1);
        auto const vth2  = vth1 * std::sqrt(T2OT1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/MaxwellianVDF-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::MaxwellianVDF::Inhomogeneous", "[LibPIC::VDF::MaxwellianVDF::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = .1, T2OT1_eq = 5.35;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1_eq, T2OT1_eq);
    auto const vdf     = MaxwellianVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

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
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta      = etas.at(i);
        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 * eta,
            desc.beta1 / 2 * desc.T2_T1 * eta * eta,
            desc.beta1 / 2 * desc.T2_T1 * eta * eta,
            0,
            0,
            0,
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(1e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(1e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(1e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta1_eq);
        auto const vth2  = vth1 * std::sqrt(T2OT1_eq * n);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/MaxwellianVDF-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::LossconeVDF::BiMax::Homogeneous", "[LibPIC::VDF::LossconeVDF::BiMax::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 5.35, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = BiMaxPlasmaDesc{ kinetic, beta1, T2OT1 };
    auto const vdf     = LossconeVDF(LossconePlasmaDesc({}, desc), geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c); // intentionally MaxwellianVDF

    CHECK(serialize(LossconePlasmaDesc({}, desc)) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 + n0_ref * Vd * Vd,
            desc.beta1 / 2 * desc.T2_T1,
            desc.beta1 / 2 * desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(1e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(1e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(1e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-7));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta1);
        auto const vth2  = vth1 * std::sqrt(T2OT1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-7));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-7));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/LossconeVDF-bi_max-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::LossconeVDF::BiMax::Inhomogeneous", "[LibPIC::VDF::LossconeVDF::BiMax::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = BiMaxPlasmaDesc(kinetic, beta1_eq, T2OT1_eq);
    auto const vdf     = LossconeVDF(LossconePlasmaDesc{ {}, desc }, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(LossconePlasmaDesc{ {}, desc }) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.0796995979174554 }.epsilon(1e-7));

    std::array const etas{
        0.4287882239954878, 0.49761722933466956, 0.5815167468245395, 0.6800561885442317, 0.788001590917828,
        0.8920754659248894, 0.9704416860618577, 1., 0.9704416860618577, 0.8920754659248894, 0.788001590917828,
        0.6800561885442317, 0.5815167468245395, 0.49761722933466956, 0.4287882239954878, 0.3733660301900688,
        0.32911727304544663, 0.29391455144617107, 0.26596447719479277, 0.24383880547267717, 0.22643378958243887,
        0.21291237930064547, 0.2026501794964986
    };
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta      = etas.at(i);
        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 * eta,
            desc.beta1 / 2 * desc.T2_T1 * eta * eta,
            desc.beta1 / 2 * desc.T2_T1 * eta * eta,
            0,
            0,
            0,
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(1e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(1e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(1e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-7));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-7));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta1_eq);
        auto const vth2  = vth1 * std::sqrt(T2OT1_eq * n);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-7));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-7));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/LossconeVDF-bi_max-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::LossconeVDF::Homogeneous", "[LibPIC::VDF::LossconeVDF::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1 = 1.5, T2OT1 = 5.35, Vd = 0;
    Real const xi = 0, D1 = 1.87, losscone_beta = 0.9, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc({ -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1);
    auto const desc    = LossconePlasmaDesc({ losscone_beta }, BiMaxPlasmaDesc{ kinetic, beta1, T2OT1 });
    auto const vdf     = LossconeVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = LossconePlasmaDesc({ losscone_beta }, { { -O0, op }, 10, ShapeOrder::CIC }, beta1 * desc.marker_temp_ratio, T2OT1 / (1 + losscone_beta));
    auto const g_vdf  = LossconeVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 + n0_ref * Vd * Vd,
            desc.beta1 / 2 * desc.T2_T1,
            desc.beta1 / 2 * desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(1e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(1e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(1e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));

        auto const n                 = *vdf.n0(ptl.pos);
        auto const vth_ratio_squared = T2OT1 / (1 + losscone_beta);
        auto const vth1              = std::sqrt(beta1);
        auto const vth2              = vth1 * std::sqrt(vth_ratio_squared);
        auto const v_mfa             = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1                = v_mfa.x - Vd;
        auto const v2                = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1))
            * (std::exp(-v2 * v2 / (vth2 * vth2)) - std::exp(-v2 * v2 / (losscone_beta * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2) / (1 - losscone_beta);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (desc.marker_temp_ratio * vth1 * vth1))
            * (std::exp(-v2 * v2 / (desc.marker_temp_ratio * vth2 * vth2)) - std::exp(-v2 * v2 / (losscone_beta * desc.marker_temp_ratio * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio)) / (1 - losscone_beta);

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/LossconeVDF-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::LossconeVDF::Inhomogeneous", "[LibPIC::VDF::LossconeVDF::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta1_eq = 1.5, T2OT1_eq = 5.35, beta_eq = .9;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = LossconePlasmaDesc({ beta_eq }, kinetic, beta1_eq, T2OT1_eq / (1 + beta_eq));
    auto const vdf     = LossconeVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta1_eq * desc.marker_temp_ratio, T2OT1_eq);
    auto const g_vdf  = LossconeVDF(LossconePlasmaDesc({ beta_eq }, g_desc), geo, { q1min, q1max - q1min }, c);

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
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta   = etas.at(i);
        auto const eta_b = eta_bs.at(i);
        auto const tmp   = (1 + beta_eq * eta_b / eta) / (1 + beta_eq) * eta;

        auto const n0_ref   = (eta - beta_eq * eta_b) / (1 - beta_eq);
        auto const nV0_ref  = Vector{};
        auto const nvv0_ref = Tensor{
            desc.beta1 / 2 * n0_ref,
            desc.beta1 / 2 * desc.T2_T1 * tmp * n0_ref,
            desc.beta1 / 2 * desc.T2_T1 * tmp * n0_ref,
            0,
            0,
            0,
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta1);
        nvv0 /= *n0 * desc.beta1;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta1);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta1);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-10));

        auto const B_div_B0          = geo.Bmag_div_B0(ptl.pos);
        auto const vth_ratio_squared = T2OT1_eq / (1 + beta_eq);
        auto const eta               = std::pow((1 - 1 / B_div_B0) * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const eta_b             = std::pow((1 - 1 / B_div_B0) * beta_eq * vth_ratio_squared + 1 / B_div_B0, -1);
        auto const beta              = beta_eq * eta_b / eta;
        auto const vth1              = std::sqrt(beta1_eq);
        auto const vth2              = vth1 * std::sqrt(vth_ratio_squared * eta);
        auto const v1                = ptl.vel.x;
        auto const v2                = std::sqrt(ptl.vel.y * ptl.vel.y + ptl.vel.z * ptl.vel.z);
        auto const f_ref
            = eta * (1 - beta) / (1 - beta_eq)
            * std::exp(-v1 * v1 / (vth1 * vth1))
            * (std::exp(-v2 * v2 / (vth2 * vth2)) - std::exp(-v2 * v2 / (beta * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2) / (1 - beta);
        auto const g_ref
            = (eta - beta_eq * eta_b) / (1 - beta_eq)
            * std::exp(-v1 * v1 / (desc.marker_temp_ratio * vth1 * vth1))
            * (std::exp(-v2 * v2 / (desc.marker_temp_ratio * vth2 * vth2)) - std::exp(-v2 * v2 / (beta * desc.marker_temp_ratio * vth2 * vth2)))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio)) / (1 - beta);

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/LossconeVDF-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::Maxwellian::Homogeneous", "[LibPIC::VDF::PartialShellVDF::Maxwellian::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 0, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, 1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c); // intentionally MaxwellianVDF

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta / 2 + n0_ref * Vd * Vd,
            desc.beta / 2 * g_desc.T2_T1,
            desc.beta / 2 * g_desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta);
        auto const vth2  = vth1 * std::sqrt(g_desc.T2_T1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-maxwellian-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::Maxwellian::Inhomogeneous", "[LibPIC::VDF::PartialShellVDF::Maxwellian::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, vs = 0, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, 1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c); // intentionally MaxwellianVDF

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta / 2 + n0_ref * Vd * Vd,
            desc.beta / 2 * g_desc.T2_T1,
            desc.beta / 2 * g_desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-10));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta);
        auto const vth2  = vth1 * std::sqrt(g_desc.T2_T1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-maxwellian-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::IsotropicShell::Homogeneous", "[LibPIC::VDF::PartialShellVDF::IsotropicShell::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = PartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta = 1;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + desc.zeta) * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const f_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const g_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-isotropic_shell-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::IsotropicShell::Inhomogeneous", "[LibPIC::VDF::PartialShellVDF::IsotropicShell::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 0;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = PartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta = 1;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + desc.zeta) * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const f_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const g_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-isotropic_shell-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::AnisotropicShell::Homogeneous", "[LibPIC::VDF::PartialShellVDF::AnisotropicShell::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 30;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = PartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta = 1;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + desc.zeta) * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-anisotropic_shell-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::PartialShellVDF::AnisotropicShell::Inhomogeneous", "[LibPIC::VDF::PartialShellVDF::AnisotropicShell::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, vs = 2;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15, zeta = 10;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = PartialShellPlasmaDesc(kinetic, beta, zeta, vs);
    auto const vdf     = PartialShellVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = PartialShellPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, zeta, vs);
    auto const g_vdf  = PartialShellVDF(g_desc, geo, { q1min, q1max - q1min }, c);

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
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta      = etas.at(i);
        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + desc.zeta) * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
            T / (3 + desc.zeta) * Real(desc.zeta + 2) / 2 * eta,
            0,
            0,
            0,
        };

        REQUIRE(vdf.n0(pos) == Scalar{ n0_ref });
        REQUIRE(geo.cart_to_mfa(vdf.nV0(pos), pos) == nV0_ref);
        REQUIRE(geo.cart_to_mfa(vdf.nvv0(pos), pos) == nvv0_ref);
    }

    // sampling
    auto const n_samples = 100000U;
    auto const particles = vdf.emit(n_samples);

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * desc.zeta) / std::tgamma(1.5 + .5 * desc.zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::pow(std::sin(alpha), zeta) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/PartialShellVDF-anisotropic_shell-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::RejectionSampling", "[LibPIC::VDF::CounterBeamVDF::RejectionSampling]")
{
    long const n_samples = 500000;
    Real const nu        = 1;

    std::vector<Real> samples(n_samples);
    std::generate_n(begin(samples), samples.size(), [nu] {
        return CounterBeamVDF::RejectionSampling{ nu }.sample();
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::Maxwellian::Homogeneous", "[LibPIC::VDF::CounterBeamVDF::Maxwellian::Homogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 10000, vs = 0, Vd = 0;
    Real const xi = 0, D1 = 1.87, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::full_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, 1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c); // intentionally MaxwellianVDF

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta / 2 + n0_ref * Vd * Vd,
            desc.beta / 2 * g_desc.T2_T1,
            desc.beta / 2 * g_desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        REQUIRE(ptl.psd.weight == Approx{ desc.initial_weight * desc.scheme + (1 - desc.scheme) * vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(2e-8));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta);
        auto const vth2  = vth1 * std::sqrt(g_desc.T2_T1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-8));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-8));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-maxwellian-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::Maxwellian::Inhomogeneous", "[LibPIC::VDF::CounterBeamVDF::Maxwellian::Inhomogeneous]")
{
    Real const O0 = 1., op = 4 * O0, c = op, beta = 1.5, nu0 = 10000, vs = 0, Vd = 0;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = BiMaxPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, 1);
    auto const g_vdf  = MaxwellianVDF(g_desc, geo, { q1min, q1max - q1min }, c); // intentionally MaxwellianVDF

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-8));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);

        auto const n0_ref   = 1;
        auto const nV0_ref  = Vector{ Vd, 0, 0 };
        auto const nvv0_ref = Tensor{
            desc.beta / 2 + n0_ref * Vd * Vd,
            desc.beta / 2 * g_desc.T2_T1,
            desc.beta / 2 * g_desc.T2_T1,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(1e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(1e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(1e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(1e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(1e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-8));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n     = *vdf.n0(ptl.pos);
        auto const vth1  = std::sqrt(beta);
        auto const vth2  = vth1 * std::sqrt(g_desc.T2_T1);
        auto const v_mfa = geo.cart_to_mfa(ptl.vel, ptl.pos);
        auto const v1    = v_mfa.x - Vd;
        auto const v2    = std::sqrt(v_mfa.y * v_mfa.y + v_mfa.z * v_mfa.z);
        auto const f_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1) - v2 * v2 / (vth2 * vth2))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2);
        auto const g_ref
            = n * std::exp(-v1 * v1 / (vth1 * vth1 * desc.marker_temp_ratio) - v2 * v2 / (vth2 * vth2 * desc.marker_temp_ratio))
            / (4 * M_PI_2 / M_2_SQRTPI * vth1 * vth2 * vth2 * desc.marker_temp_ratio * std::sqrt(desc.marker_temp_ratio));

        REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-8));
        REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-8));
    }

    if constexpr (dump_samples) {
        static_assert(n_samples > 0);
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-maxwellian-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::IsotropicShell::Homogeneous", "[LibPIC::VDF::CounterBeamVDF::IsotropicShell::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 10000, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = CounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-10));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta  = 1;
        auto const       zeta = 0;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + zeta) * eta,
            T / (3 + zeta) * Real(zeta + 2) / 2 * eta,
            T / (3 + zeta) * Real(zeta + 2) / 2 * eta,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(4e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(1e-1));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const zeta = 0;
        auto const n    = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const f_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-8));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const g_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-8));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-isotropic_shell-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::IsotropicShell::Inhomogeneous", "[LibPIC::VDF::CounterBeamVDF::IsotropicShell::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 10000, vs = 10;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, ParticleScheme::delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = CounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 1.0 / (q1max - q1min) }.epsilon(1e-8));

    for (long q1 = q1min; q1 <= q1max; ++q1) {
        CurviCoord const pos(q1);
        auto const       eta  = 1;
        auto const       zeta = 0;

        auto const n0_ref   = eta;
        auto const nV0_ref  = Vector{};
        auto const xs       = desc.vs / std::sqrt(beta);
        auto const Ab       = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
        auto const T        = .5 * desc.beta / Ab * (xs * (2.5 + xs * xs) * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.75 + xs * xs * (3 + xs * xs)) * std::erfc(-xs));
        auto const nvv0_ref = Tensor{
            T / (3 + zeta) * eta,
            T / (3 + zeta) * Real(zeta + 2) / 2 * eta,
            T / (3 + zeta) * Real(zeta + 2) / 2 * eta,
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(4e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(3e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-1));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-14));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const zeta = 0;
        auto const n    = *vdf.n0(ptl.pos);
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const f_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-8));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bz    = 2 / M_2_SQRTPI * std::tgamma(1 + .5 * zeta) / std::tgamma(1.5 + .5 * zeta);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const g_ref = n * std::exp(-v * v / vth2) / (M_PI * 2 * vth2 * std::sqrt(vth2) * Ab * Bz);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-8));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-isotropic_shell-inhomogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::CounterBeam::Homogeneous", "[LibPIC::VDF::CounterBeamVDF::CounterBeam::Homogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.1, vs = 10;
    Real const xi = 0, D1 = 1, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = CounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(2e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(2e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(3e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n  = *vdf.n0(ptl.pos);
        auto const nu = std::sqrt(geo.Bmag_div_B0(ptl.pos)) * nu0;
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-counter_beam-homogeneous.m" };
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

TEST_CASE("Test LibPIC::VDF::CounterBeamVDF::CounterBeam::Inhomogeneous", "[LibPIC::VDF::CounterBeamVDF::CounterBeam::Inhomogeneous]")
{
    Real const O0 = 1, op = 4 * O0, c = op, beta = 1.5, nu0 = 0.2, vs = 2;
    Real const xi = .876, xiD1q1max = M_PI_2 * 0.8, psd_refresh_frequency = 0;
    long const q1min = -7, q1max = 15;
    auto const D1      = xiD1q1max / (xi * std::max(std::abs(q1min), std::abs(q1max)));
    auto const geo     = Geometry{ xi, D1, O0 };
    auto const kinetic = KineticPlasmaDesc{ { -O0, op }, 10, ShapeOrder::CIC, psd_refresh_frequency, delta_f, .001, 2.1 };
    auto const desc    = CounterBeamPlasmaDesc(kinetic, beta, nu0, vs);
    auto const vdf     = CounterBeamVDF(desc, geo, { q1min, q1max - q1min }, c);

    auto const g_desc = CounterBeamPlasmaDesc({ { -O0, op }, 10, ShapeOrder::CIC }, beta * desc.marker_temp_ratio, nu0, vs);
    auto const g_vdf  = CounterBeamVDF(g_desc, geo, { q1min, q1max - q1min }, c);

    CHECK(serialize(desc) == serialize(vdf.plasma_desc()));

    // check equilibrium macro variables
    CHECK(vdf.Nrefcell_div_Ntotal() == Approx{ 0.02072000799696996 }.epsilon(1e-8));

    std::array const etas{
        1.4561623198207398, 1.3112428726471927, 1.2036232244145273,
        1.1243388313063896, 1.0675036398265505, 1.0292671327256642,
        1.007210295647286, 1., 1.007210295647286, 1.0292671327256642,
        1.0675036398265505, 1.1243388313063896, 1.2036232244145273,
        1.3112428726471927, 1.4561623198207398, 1.6522706092531634,
        1.9218119442765296, 2.30223983358348, 2.8613029050290333,
        3.733726751769592, 5.213117245225659, 7.948425777815061,
        13.162564632484335
    };
    for (unsigned i = 0; i < std::size(etas); ++i) {
        auto const q1  = q1min + i;
        auto const pos = CurviCoord(q1);

        auto const eta      = etas.at(i);
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

    // moments
    if constexpr (enable_moment_checks) {
        auto n0   = Scalar{};
        auto nV0  = CartVector{};
        auto nvv0 = CartTensor{};
        for (long i = q1min; i < q1max; ++i) {
            auto const pos = CurviCoord{ i + 0.5 };

            n0 += vdf.n0(pos);
            nV0 += vdf.nV0(pos);
            nvv0 += vdf.nvv0(pos);
        }
        nV0 /= *n0 * std::sqrt(desc.beta);
        nvv0 /= *n0 * desc.beta;
        n0 /= *n0;

        auto const particle_density = std::accumulate(
            begin(particles), end(particles), Real{}, [&](Real const sum, Particle const &ptl) {
                return sum + 1 * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_density == Approx{ n0 }.epsilon(1e-2));

        auto const particle_flux = std::accumulate(
            begin(particles), end(particles), CartVector{}, [&](CartVector const &sum, Particle const &ptl) {
                auto const vel = ptl.vel / std::sqrt(desc.beta);
                return sum + vel * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(particle_flux.x == Approx{ nV0.x }.margin(2e-2));
        CHECK(particle_flux.y == Approx{ nV0.y }.margin(2e-2));
        CHECK(particle_flux.z == Approx{ nV0.z }.margin(2e-2));

        auto const stress_energy = std::accumulate(
            begin(particles), end(particles), CartTensor{}, [&](CartTensor const &sum, Particle const &ptl) {
                auto const v  = ptl.vel / std::sqrt(desc.beta);
                auto const vv = CartTensor{ v.x * v.x, v.y * v.y, v.z * v.z, v.x * v.y, v.y * v.z, v.z * v.x };
                return sum + vv * (ptl.psd.real_f / ptl.psd.marker + desc.scheme * ptl.psd.weight) / n_samples;
            });
        CHECK(stress_energy.xx == Approx{ nvv0.xx }.epsilon(2e-2));
        CHECK(stress_energy.yy == Approx{ nvv0.yy }.epsilon(3e-2));
        CHECK(stress_energy.zz == Approx{ nvv0.zz }.epsilon(4e-2));
        CHECK(stress_energy.xy == Approx{ nvv0.xy }.margin(2e-2));
        CHECK(stress_energy.yz == Approx{ nvv0.yz }.margin(2e-2));
        CHECK(stress_energy.zx == Approx{ nvv0.zx }.margin(2e-2));
    }

    static_assert(n_samples > 100);
    for (unsigned long i = 0; i < 100; ++i) {
        Particle const &ptl = particles[i];

        if (full_f == desc.scheme)
            REQUIRE(ptl.psd.weight == Approx{ vdf.f0(ptl) / g_vdf.f0(ptl) }.margin(1e-14));
        else
            REQUIRE(std::abs(ptl.psd.weight) < desc.initial_weight);
        REQUIRE(ptl.psd.marker == Approx{ g_vdf.f0(ptl) }.epsilon(1e-13));
        REQUIRE(ptl.psd.real_f == Approx{ vdf.f0(ptl) * desc.scheme + ptl.psd.weight * ptl.psd.marker }.epsilon(1e-14));
        REQUIRE(vdf.real_f0(ptl) == Approx{ vdf.f0(ptl) }.epsilon(1e-14));

        auto const n  = *vdf.n0(ptl.pos);
        auto const nu = std::sqrt(geo.Bmag_div_B0(ptl.pos)) * nu0;
        {
            auto const vth2  = beta;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const f_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.f0(ptl) == Approx{ f_ref }.epsilon(1e-10));
        }
        {
            auto const vth2  = beta * desc.marker_temp_ratio;
            auto const xs    = desc.vs / std::sqrt(vth2);
            auto const Ab    = .5 * (xs * std::exp(-xs * xs) + 2 / M_2_SQRTPI * (.5 + xs * xs) * std::erfc(-xs));
            auto const Bnu   = 2 * nu * Faddeeva::Dawson(1 / nu);
            auto const v     = std::sqrt(dot(ptl.vel, ptl.vel)) - desc.vs;
            auto const alpha = std::acos(ptl.vel.x / (v + desc.vs));
            auto const g_ref = n * std::exp(-v * v / vth2) * std::exp(-std::pow(std::sin(alpha), 2) / (nu * nu)) / (2 * M_PI * vth2 * std::sqrt(vth2) * Ab * Bnu);
            REQUIRE(vdf.g0(ptl) == Approx{ g_ref }.epsilon(1e-10));
        }
    }

    if constexpr (dump_samples) {
        std::ofstream os{ "/Users/kyungguk/Downloads/CounterBeamVDF-counter_beam-inhomogeneous.m" };
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
