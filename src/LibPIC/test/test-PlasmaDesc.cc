/*
 * Copyright (c) 2021-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/PlasmaDesc.h>
#include <exception>
#include <limits>

namespace {
template <class T, class U>
[[nodiscard]] bool operator==(Detail::VectorTemplate<T, double> const &a, Detail::VectorTemplate<U, double> const &b) noexcept
{
    return a.x == Approx{ b.x }.margin(1e-15)
        && a.y == Approx{ b.y }.margin(1e-15)
        && a.z == Approx{ b.z }.margin(1e-15);
}
[[nodiscard]] bool operator==(CurviCoord const &a, CurviCoord const &b)
{
    return a.q1 == b.q1;
}
[[nodiscard]] constexpr bool operator==(std::complex<double> const &a, std::complex<double> const &b) noexcept
{
    return a.real() == b.real() && a.imag() == b.imag();
}
} // namespace
using ::operator==;

TEST_CASE("Test LibPIC::PlasmaDesc", "[LibPIC::PlasmaDesc]")
{
    constexpr auto desc1 = PlasmaDesc{ 1, 2, 3 };
    CHECK(desc1.Oc == 1);
    CHECK(desc1.op == 2);
    CHECK(desc1.number_of_source_smoothings == 3);

    constexpr auto desc2 = PlasmaDesc{ 1, 2 };
    CHECK(desc2.Oc == 1);
    CHECK(desc2.op == 2);
    CHECK(desc2.number_of_source_smoothings == 0);

    constexpr auto s1 = serialize(desc1);
    constexpr auto s2 = serialize(desc2);
    CHECK(s1 == s2);
    CHECK(desc1 == desc2);
    CHECK(std::get<0>(s1) == desc1.Oc);
    CHECK(std::get<1>(s1) == desc1.op);

    CHECK_THROWS_AS(PlasmaDesc(0, 1), std::exception);
    CHECK_THROWS_AS(PlasmaDesc(-1, 0), std::exception);
    CHECK_THROWS_AS(PlasmaDesc(-1, -1), std::exception);

    constexpr auto quiet_nan = std::numeric_limits<Real>::quiet_NaN();
    CHECK_NOTHROW(PlasmaDesc(1, quiet_nan));
    CHECK_NOTHROW(PlasmaDesc(quiet_nan, 1));
}

TEST_CASE("Test LibPIC::eFluidDesc", "[LibPIC::eFluidDesc]")
{
    constexpr auto base1 = PlasmaDesc{ 1, 2, 3 };
    constexpr auto desc1 = eFluidDesc(base1, 1.1, Closure::isothermal);
    CHECK(desc1 == base1);
    CHECK(desc1.beta == 1.1);
    CHECK(desc1.gamma == 1);

    constexpr auto s1 = serialize(desc1);
    CHECK(std::get<2>(s1) == desc1.beta);
    CHECK(std::get<3>(s1) == desc1.gamma);

    constexpr auto base2 = PlasmaDesc{ 1, 2 };
    constexpr auto desc2 = eFluidDesc(base2);
    CHECK(desc2 == base2);
    CHECK(desc2.beta == 0);
    CHECK(desc2.gamma == 5. / 3);

    CHECK_THROWS_AS(eFluidDesc(base1, -1), std::exception);
}

TEST_CASE("Test LibPIC::ColdPlasmaDesc", "[LibPIC::ColdPlasmaDesc]")
{
    constexpr auto base1 = PlasmaDesc{ 1, 2, 3 };
    constexpr auto desc1 = ColdPlasmaDesc(base1);
    CHECK(desc1 == base1);

    constexpr auto base2 = PlasmaDesc{ 1, 2 };
    constexpr auto desc2 = ColdPlasmaDesc(base2);
    CHECK(desc2 == base2);
}

TEST_CASE("Test LibPIC::KineticPlasmaDesc", "[LibPIC::KineticPlasmaDesc]")
{
    constexpr auto base1 = PlasmaDesc{ 1, 2, 3 };
    constexpr auto desc1 = KineticPlasmaDesc(base1, 100, ShapeOrder::CIC);
    CHECK(desc1 == base1);
    CHECK(desc1.Nc == 100);
    CHECK(desc1.shape_order == 1);
    CHECK(desc1.psd_refresh_frequency == 0);
    CHECK(desc1.should_refresh_psd == false);
    CHECK(desc1.scheme == ParticleScheme::full_f);
    CHECK(desc1.initial_weight == 0);
    CHECK(desc1.marker_temp_ratio == 1);

    constexpr auto base2 = PlasmaDesc{ 1, 2 };
    constexpr auto desc2
        = KineticPlasmaDesc(base2, 10, ShapeOrder::_3rd, .5, ParticleScheme::delta_f, .1, 2);
    CHECK(desc2 == base2);
    CHECK(desc2.shape_order == ShapeOrder::_3rd);
    CHECK(desc2.psd_refresh_frequency == .5);
    CHECK(desc2.should_refresh_psd == true);
    CHECK(desc2.scheme == ParticleScheme::delta_f);
    CHECK(desc2.initial_weight == .1);
    CHECK(desc2.marker_temp_ratio == 2);

    constexpr auto s1 = serialize(desc1);
    CHECK(std::get<2>(s1) == desc1.Nc);
    CHECK(std::get<3>(s1) == desc1.scheme);
    CHECK(desc1 == desc1);

    CHECK_THROWS_AS(KineticPlasmaDesc(base1, 0, ShapeOrder::TSC), std::exception);
    CHECK_THROWS_AS(KineticPlasmaDesc(base1, 0, ShapeOrder::TSC, -1), std::exception);
    CHECK_THROWS_AS(KineticPlasmaDesc(base1, 0, ShapeOrder::TSC, 0, ParticleScheme::delta_f, -1), std::exception);
}

TEST_CASE("Test LibPIC::TestParticleDesc", "[LibPIC::TestParticleDesc]")
{
    constexpr unsigned                      Nptls = 2;
    constexpr std::array<MFAVector, Nptls>  vel   = { MFAVector{ 1, 2, 3 }, { 4, 5, 6 } };
    constexpr std::array<CurviCoord, Nptls> pos   = { CurviCoord{ 1 }, CurviCoord{ 2 } };
    constexpr auto                          desc1 = TestParticleDesc<Nptls>({ 1, 2, 3 }, vel, pos);
    CHECK(desc1.op == 0);
    CHECK(desc1.Nc == Nptls);
    CHECK(desc1.number_of_source_smoothings == 0);
    CHECK(desc1.number_of_test_particles == Nptls);
    CHECK(desc1.vel == vel);
    CHECK(desc1.pos == pos);

    (void)serialize(desc1);
    CHECK(desc1 == desc1);
}

TEST_CASE("Test LibPIC::BiMaxPlasmaDesc", "[LibPIC::BiMaxPlasmaDesc]")
{
    constexpr auto base1 = KineticPlasmaDesc{ { 1, 2, 3 }, 100, ShapeOrder::CIC };
    constexpr auto desc1 = BiMaxPlasmaDesc(base1, 1, 2);
    CHECK(desc1 == base1);
    CHECK(desc1.beta1 == 1);
    CHECK(desc1.T2_T1 == 2);

    constexpr auto base2 = KineticPlasmaDesc{ { 1, 2 }, 10, ShapeOrder::_3rd, ParticleScheme::delta_f };
    constexpr auto desc2 = BiMaxPlasmaDesc(base2, .1);
    CHECK(desc2 == base2);
    CHECK(desc2.beta1 == .1);
    CHECK(desc2.T2_T1 == 1);

    constexpr auto s1 = serialize(desc1);
    CHECK(std::get<6>(s1) == desc1.beta1);
    CHECK(std::get<7>(s1) == desc1.T2_T1);
    CHECK(desc1 == desc1);

    CHECK_THROWS_AS(BiMaxPlasmaDesc(base1, 0), std::exception);
    CHECK_THROWS_AS(BiMaxPlasmaDesc(base1, -1), std::exception);
}

TEST_CASE("Test LibPIC::LossconePlasmaDesc", "[LibPIC::LossconePlasmaDesc]")
{
    constexpr auto base1 = BiMaxPlasmaDesc({ { 1, 2, 3 }, 100, ShapeOrder::CIC }, 1, 2);
    constexpr auto desc1 = LossconePlasmaDesc({ .3 }, base1);
    CHECK(desc1 == base1);
    CHECK(desc1.losscone.beta == .3);
    constexpr auto desc2 = LossconePlasmaDesc({}, base1);
    CHECK(desc2 == base1);
    CHECK(desc2.losscone.beta == 0);
    CHECK_THROWS_AS(LossconePlasmaDesc({ -1 }, base1), std::exception);

    constexpr auto base2 = static_cast<KineticPlasmaDesc const &>(base1);
    constexpr auto desc3 = LossconePlasmaDesc({}, base2, base1.beta1);
    CHECK(desc3 == base2);
    CHECK(desc3.beta1 == 1);
    CHECK(desc3.T2_T1 == 1);
    CHECK(desc3.losscone.beta == 0);
    constexpr auto desc4 = LossconePlasmaDesc({}, base2, base1.beta1, 2);
    CHECK(desc4 == base2);
    CHECK(desc4.beta1 == 1);
    CHECK(desc4.T2_T1 == 2);
    CHECK(desc4.losscone.beta == 0);
    constexpr auto desc5 = LossconePlasmaDesc({ .1 }, base2, base1.beta1, 2);
    CHECK(desc5 == base2);
    CHECK(desc5.beta1 == 1);
    CHECK(desc5.T2_T1 == 2 * (1 + .1));
    CHECK(desc5.losscone.beta == .1);
    constexpr auto desc6 = LossconePlasmaDesc({ 0 }, base2, base1.beta1, base1.T2_T1);
    CHECK(desc6 == base1);

    CHECK_THROWS_AS(LossconePlasmaDesc({}, base2, .1, -1), std::exception);
    CHECK_THROWS_AS(LossconePlasmaDesc({ -1 }, base2, .1, 1), std::exception);
}

TEST_CASE("Test LibPIC::PartialShellPlasmaDesc", "[LibPIC::PartialShellPlasmaDesc]")
{
    constexpr auto base1 = KineticPlasmaDesc{ { 1, 2, 3 }, 100, ShapeOrder::CIC };
    constexpr auto desc1 = PartialShellPlasmaDesc(base1, 1, 2, 3);
    CHECK(desc1 == base1);
    CHECK(desc1.beta == 1);
    CHECK(desc1.zeta == 2);
    CHECK(desc1.vs == 3);

    constexpr auto base2 = KineticPlasmaDesc{ { 1, 2 }, 10, ShapeOrder::_3rd, ParticleScheme::delta_f };
    constexpr auto desc2 = PartialShellPlasmaDesc(base2, .1);
    CHECK(desc2 == base2);
    CHECK(desc2.beta == .1);
    CHECK(desc2.zeta == 0);
    CHECK(desc2.vs == 0);

    constexpr auto s1 = serialize(desc1);
    CHECK(std::get<6>(s1) == desc1.beta);
    CHECK(std::get<7>(s1) == desc1.zeta);
    CHECK(std::get<8>(s1) == desc1.vs);
    CHECK(desc1 == desc1);

    CHECK_THROWS_AS(PartialShellPlasmaDesc(base1, 0), std::exception);
    CHECK_THROWS_AS(PartialShellPlasmaDesc(base1, -1), std::exception);
    CHECK_THROWS_AS(PartialShellPlasmaDesc(base1, 1, 0, -1), std::exception);
}

TEST_CASE("Test LibPIC::CounterBeamPlasmaDesc", "[LibPIC::CounterBeamPlasmaDesc]")
{
    constexpr auto base1 = KineticPlasmaDesc{ { 1, 2, 3 }, 100, ShapeOrder::CIC };
    constexpr auto desc1 = CounterBeamPlasmaDesc(base1, 1, 2, 3);
    CHECK(desc1 == base1);
    CHECK(desc1.beta == 1);
    CHECK(desc1.nu == 2);
    CHECK(desc1.vs == 3);

    constexpr auto base2 = KineticPlasmaDesc{ { 1, 2 }, 10, ShapeOrder::_3rd, ParticleScheme::delta_f };
    constexpr auto desc2 = CounterBeamPlasmaDesc(base2, .1, 1.3);
    CHECK(desc2 == base2);
    CHECK(desc2.beta == .1);
    CHECK(desc2.nu == 1.3);
    CHECK(desc2.vs == 0);

    constexpr auto s1 = serialize(desc1);
    CHECK(std::get<6>(s1) == desc1.beta);
    CHECK(std::get<7>(s1) == desc1.nu);
    CHECK(std::get<8>(s1) == desc1.vs);
    CHECK(desc1 == desc1);

    CHECK_THROWS_AS(CounterBeamPlasmaDesc(base1, 0, 1), std::exception);
    CHECK_THROWS_AS(CounterBeamPlasmaDesc(base1, -1, 1), std::exception);
    CHECK_THROWS_AS(CounterBeamPlasmaDesc(base1, 1, 0), std::exception);
    CHECK_THROWS_AS(CounterBeamPlasmaDesc(base1, 1, -1), std::exception);
    CHECK_THROWS_AS(CounterBeamPlasmaDesc(base1, 1, 0, -1), std::exception);
}

TEST_CASE("Test LibPIC::ExternalSourceDesc", "[LibPIC::ExternalSourceDesc]")
{
    CHECK_THROWS_AS(ExternalSourceBase(-1, { 0, -1 }, -1), std::invalid_argument);
    CHECK_THROWS_AS(ExternalSourceBase(-1, { 0, -1 }, { 1, -1 }), std::invalid_argument);

    {
        constexpr auto N    = 2U;
        constexpr auto desc = ExternalSourceDesc<N>{
            { 1, { 1, 10 }, 2, 3 },
            { ComplexVector{ { 1., 1 }, 2., { 3., 3 } }, { 1i, .5i, 1 } },
            { CurviCoord{ -1 }, CurviCoord{ 1 } }
        };

        constexpr auto s1 = serialize(desc);
        CHECK(std::get<0>(s1) == desc.number_of_source_smoothings);
        CHECK(std::get<1>(s1) == desc.omega);
        CHECK(std::get<2>(s1) == desc.extent.loc);
        CHECK(std::get<3>(s1) == desc.extent.len);
        CHECK(std::get<4>(s1) == desc.ease.in);
        CHECK(std::get<5>(s1) == desc.ease.out);
        CHECK(std::get<6>(s1) == desc.number_of_source_points);

        CHECK(!std::isfinite(desc.Oc));
        CHECK(!std::isfinite(desc.op));
        CHECK(desc.number_of_source_smoothings == 3);
        CHECK(desc.omega == 1);
        CHECK(desc.extent.loc == 1);
        CHECK(desc.extent.len == 10);
        CHECK(desc.ease.in == 2);
        CHECK(desc.ease.out == 2);
        CHECK(desc.number_of_source_points == N);
        CHECK(std::get<0>(desc.pos).q1 == -1);
        CHECK(std::get<1>(desc.pos).q1 == +1);
        CHECK(std::get<0>(desc.J0).x == 1. + 1i);
        CHECK(std::get<0>(desc.J0).y == 2. + 0i);
        CHECK(std::get<0>(desc.J0).z == 3. + 3i);
        CHECK(std::get<1>(desc.J0).x == 0. + 1i);
        CHECK(std::get<1>(desc.J0).y == 0. + .5i);
        CHECK(std::get<1>(desc.J0).z == 1. + 0i);
    }
    {
        constexpr auto N    = 2U;
        constexpr auto desc = ExternalSourceDesc<N>{
            { 1, { 1, 10 }, { 3, 2 }, 3 },
            { ComplexVector{ { 1., 1 }, 2., { 3., 3 } }, { 1i, .5i, 1 } },
            { CurviCoord{ -1 }, CurviCoord{ 1 } }
        };

        constexpr auto s1 = serialize(desc);
        CHECK(std::get<0>(s1) == desc.number_of_source_smoothings);
        CHECK(std::get<1>(s1) == desc.omega);
        CHECK(std::get<2>(s1) == desc.extent.loc);
        CHECK(std::get<3>(s1) == desc.extent.len);
        CHECK(std::get<4>(s1) == desc.ease.in);
        CHECK(std::get<5>(s1) == desc.ease.out);
        CHECK(std::get<5>(s1) == desc.number_of_source_points);

        CHECK(!std::isfinite(desc.Oc));
        CHECK(!std::isfinite(desc.op));
        CHECK(desc.number_of_source_smoothings == 3);
        CHECK(desc.omega == 1);
        CHECK(desc.extent.loc == 1);
        CHECK(desc.extent.len == 10);
        CHECK(desc.ease.in == 3);
        CHECK(desc.ease.out == 2);
        CHECK(desc.number_of_source_points == N);
        CHECK(std::get<0>(desc.pos).q1 == -1);
        CHECK(std::get<1>(desc.pos).q1 == +1);
        CHECK(std::get<0>(desc.J0).x == 1. + 1i);
        CHECK(std::get<0>(desc.J0).y == 2. + 0i);
        CHECK(std::get<0>(desc.J0).z == 3. + 3i);
        CHECK(std::get<1>(desc.J0).x == 0. + 1i);
        CHECK(std::get<1>(desc.J0).y == 0. + .5i);
        CHECK(std::get<1>(desc.J0).z == 1. + 0i);
    }
}
