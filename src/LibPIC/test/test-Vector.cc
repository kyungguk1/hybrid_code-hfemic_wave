/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/VT/ComplexVector.h>
#include <PIC/VT/FourVector.h>
#include <PIC/VT/Vector.h>

namespace {
template <class T, class U>
[[nodiscard]] bool operator==(Detail::VectorTemplate<T, double> const &a, Detail::VectorTemplate<U, double> const &b) noexcept
{
    return a.x == Approx{ b.x }.margin(1e-14)
        && a.y == Approx{ b.y }.margin(1e-14)
        && a.z == Approx{ b.z }.margin(1e-14);
}
template <class T1, class T2, class U1, class U2>
[[nodiscard]] bool operator==(Detail::FourVectorTemplate<T1, T2> const &a, Detail::FourVectorTemplate<U1, U2> const &b) noexcept
{
    return a.t == Approx{ b.t }.margin(1e-14) && a.s == b.s;
}
} // namespace
using ::operator==;
using std::get;

TEST_CASE("Test LibPIC::Vector", "[LibPIC::Vector]")
{
    {
        constexpr Vector v1{};
        constexpr bool   tf = v1.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        });
        CHECK(tf);

        constexpr Vector v2{ 1 };
        CHECK(v2.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 1;
        }));

        constexpr Vector v3 = [](Vector v) {
            get<0>(v) = 1;
            get<1>(v) = 2;
            get<2>(v) = 3;
            return v;
        }({});
        static_assert(get<0>(v3) == 1);
        static_assert(get<1>(v3) == 2);
        static_assert(get<2>(v3) == 3);
        CHECK((v3.x == 1 && v3.y == 2 && v3.z == 3));
        constexpr auto tmp = Vector{ 1, 2, 3 };
        CHECK(v3 == tmp);

        constexpr bool tf2 = std::addressof(v1) == std::addressof(+v1);
        CHECK(tf2);

        constexpr Vector v4 = -v3;
        CHECK((v4.x == -1 && v4.y == -2 && v4.z == -3));

        constexpr auto dot1 = dot(v1, v3);
        CHECK(dot1 == 0);
        constexpr auto dot2 = dot(v2, v3);
        CHECK(dot2 == 6);

        constexpr auto cross1 = cross(v1, v3);
        CHECK(cross1.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        constexpr auto cross2 = cross(Vector{ 4, 3, 8 }, v3);
        CHECK((cross2.x == -7 && cross2.y == -4 && cross2.z == 5));
    }

    {
        constexpr auto is_equal = [](Vector lhs, Vector rhs) {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
        };

        constexpr Vector v1{ 1, 2, 3 };
        constexpr double x{ 1 };
        CHECK(is_equal(v1 + x, { 2, 3, 4 }));
        CHECK(is_equal(v1 - x, { 0, 1, 2 }));
        CHECK(is_equal(v1 * x, v1));
        CHECK(is_equal(v1 / x, v1));
        CHECK(is_equal(x + v1, v1 + x));
        CHECK(is_equal(x - v1, -(v1 - x)));
        CHECK(is_equal(x * v1, v1));
        CHECK(is_equal(x / v1, { x / 1, x / 2, x / 3 }));

        constexpr Vector v2 = v1 * 10.;
        CHECK(is_equal(v1 + v2, { 11, 22, 33 }));
        CHECK(is_equal(v2 - v1, { 9, 18, 27 }));
        CHECK(is_equal(v1 * v2, { 10, 40, 90 }));
        CHECK(is_equal(v2 / v1, { 10, 10, 10 }));
    }
}

TEST_CASE("Test LibPIC::ComplexVector", "[LibPIC::ComplexVector]")
{
    using Vector = ComplexVector;
    {
        constexpr Vector v1{};
        constexpr bool   tf = v1.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0.;
        });
        CHECK(tf);

        constexpr Vector v2{ 1 };
        CHECK(v2.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 1.;
        }));

        Vector const v3 = [](Vector v) {
            get<0>(v) = 1.;
            get<1>(v) = 2.;
            get<2>(v) = 3.;
            return v;
        }({});
        CHECK(get<0>(v3) == 1.);
        CHECK(get<1>(v3) == 2.);
        CHECK(get<2>(v3) == 3.);
        CHECK((v3.x == 1. && v3.y == 2. && v3.z == 3.));

        constexpr bool tf2 = std::addressof(v1) == std::addressof(+v1);
        CHECK(tf2);

        Vector const v4 = -v3;
        CHECK((v4.x == -1. && v4.y == -2. && v4.z == -3.));

        auto const dot1 = dot(v1, v3);
        CHECK(dot1 == 0.);
        auto const dot2 = dot(v2, v3);
        CHECK(dot2 == 6.);

        auto const cross1 = cross(v1, v3);
        CHECK(cross1.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0.;
        }));
        auto const cross2 = cross(Vector{ 4, 3, 8 }, v3);
        CHECK((cross2.x == -7. && cross2.y == -4. && cross2.z == 5.));
    }

    {
        auto const is_equal = [](Vector lhs, Vector rhs) {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
        };

        constexpr Vector v1{ 1, 2, 3 };
        constexpr double x{ 1 };
        CHECK(is_equal(v1 + x, { 2, 3, 4 }));
        CHECK(is_equal(v1 - x, { 0, 1, 2 }));
        CHECK(is_equal(v1 * x, v1));
        CHECK(is_equal(v1 / x, v1));
        CHECK(is_equal(x + v1, v1 + x));
        CHECK(is_equal(x - v1, -(v1 - x)));
        CHECK(is_equal(x * v1, v1));
        CHECK(is_equal(x / v1, { x / 1, x / 2, x / 3 }));

        Vector const v2 = v1 * 10.;
        CHECK(is_equal(v1 + v2, { 11, 22, 33 }));
        CHECK(is_equal(v2 - v1, { 9, 18, 27 }));
        CHECK(is_equal(v1 * v2, { 10, 40, 90 }));
        CHECK(is_equal(v2 / v1, { 10, 10, 10 }));
    }
}

TEST_CASE("Test LibPIC::SpecialVectors", "[LibPIC::SpecialVectors]")
{
    constexpr auto vector = Vector{ 1, 2, 3 };
    {
        constexpr auto cart = CartVector{ vector };
        CHECK(vector == cart);

        constexpr auto mfa = MFAVector{ vector };
        CHECK(vector == mfa);

        constexpr auto covar = CovarVector{ vector };
        CHECK(vector == covar);

        constexpr auto contr = ContrVector{ vector };
        CHECK(vector == contr);
    }
    {
        constexpr auto scale_factors = Vector{ .3, 3, .9384 };
        constexpr auto covar         = CovarVector{ vector * scale_factors };
        constexpr auto contr         = ContrVector{ vector / scale_factors };
        CHECK(dot(covar, contr) == dot(vector, vector));
        CHECK(dot(contr, covar) == dot(vector, vector));
        CHECK(cross(covar, covar, scale_factors.fold(1., std::multiplies{})) * ContrVector(scale_factors) == cross(vector, vector));
        CHECK(cross(contr, contr, scale_factors.fold(1., std::multiplies{})) * CovarVector(scale_factors) == cross(vector, vector));
    }
}

TEST_CASE("Test LibPIC::FourVector", "[LibPIC::FourVector]")
{
    {
        constexpr FourVector v1{};
        CHECK((*v1.t == 0 && v1.s.x == 0 && v1.s.y == 0 && v1.s.z == 0));

        constexpr FourVector v2{ 1 };
        CHECK((*v2.t == 1 && v2.s.x == 1 && v2.s.y == 1 && v2.s.z == 1));

        constexpr FourVector v3 = [](FourVector fv) {
            get<0>(fv) = 1;
            get<1>(fv) = 2;
            get<2>(fv) = 3;
            get<3>(fv) = 4;
            return fv;
        }({});
        static_assert(*get<0>(v3) == 1);
        static_assert(get<1>(v3) == 2);
        static_assert(get<2>(v3) == 3);
        static_assert(get<3>(v3) == 4);
        CHECK((*v3.t == 1 && v3.s.x == 2 && v3.s.y == 3 && v3.s.z == 4));
        constexpr auto tmp = FourVector{ 1, { 2, 3, 4 } };
        CHECK(v3 == tmp);

        constexpr bool tf2 = std::addressof(v1) == std::addressof(+v1);
        CHECK(tf2);

        constexpr FourVector v4 = -v3;
        CHECK((*v4.t == -1 && v4.s.x == -2 && v4.s.y == -3 && v4.s.z == -4));
    }

    {
        constexpr auto is_equal = [](FourVector const &lhs, FourVector const &rhs) {
            return *lhs.t == *rhs.t && lhs.s.x == rhs.s.x && lhs.s.y == rhs.s.y
                && lhs.s.z == rhs.s.z;
        };

        constexpr FourVector v1{ -1, { 1, 2, 3 } };
        constexpr double     x{ 1 };
        CHECK(is_equal(v1 + x, { 0, { 2, 3, 4 } }));
        CHECK(is_equal(v1 - x, { -2, { 0, 1, 2 } }));
        CHECK(is_equal(v1 * x, v1));
        CHECK(is_equal(v1 / x, v1));
        CHECK(is_equal(x + v1, v1 + x));
        CHECK(is_equal(x - v1, -(v1 - x)));
        CHECK(is_equal(x * v1, v1));
        CHECK(is_equal(x / v1, { x / -1, { x / 1, x / 2, x / 3 } }));

        constexpr FourVector v2 = v1 * 10.;
        CHECK(is_equal(v1 + v2, { -11, { 11, 22, 33 } }));
        CHECK(is_equal(v2 - v1, { -9, { 9, 18, 27 } }));
        CHECK(is_equal(v1 * v2, { 10, { 10, 40, 90 } }));
        CHECK(is_equal(v2 / v1, { 10, { 10, 10, 10 } }));
    }

    {
        constexpr auto beta     = .8;
        constexpr auto gamma    = 1.666666666666667; // 1 / std::sqrt((1 - beta) * (1 + beta));
        constexpr auto normal   = Vector{ 0.4364357804719848, -0.8728715609439696, 0.2182178902359924 };
        constexpr auto unprimed = FourVector{ 1.1, { .3, -1.4, 1.3 } };
        FourVector     primed;

        primed = lorentz_boost<1>(unprimed, gamma * beta);
        CHECK(primed == FourVector(1.4333333333333336, { -0.966666666666667, -1.4, 1.3 }));
        CHECK(unprimed == lorentz_boost<1>(primed, -gamma * beta));
        primed = lorentz_boost<-1>(unprimed, -gamma * beta);
        CHECK(primed == FourVector(1.4333333333333336, { -0.966666666666667, -1.4, 1.3 }));
        CHECK(unprimed == lorentz_boost<-1>(primed, gamma * beta));
        {
            auto const tmp1 = lorentz_boost<1>(unprimed, beta, gamma);
            CHECK(tmp1 == primed);
            auto const tmp2 = lorentz_boost<-1>(tmp1, beta, gamma);
            CHECK(tmp2 == unprimed);
        }

        primed = lorentz_boost<2>(unprimed, gamma * beta);
        CHECK(primed == FourVector(3.7, { 0.3, -3.8000000000000007, 1.3 }));
        CHECK(unprimed == lorentz_boost<2>(primed, -gamma * beta));
        primed = lorentz_boost<-2>(unprimed, -gamma * beta);
        CHECK(primed == FourVector(3.7, { 0.3, -3.8000000000000007, 1.3 }));
        CHECK(unprimed == lorentz_boost<-2>(primed, gamma * beta));
        {
            auto const tmp1 = lorentz_boost<2>(unprimed, beta, gamma);
            CHECK(tmp1 == primed);
            auto const tmp2 = lorentz_boost<-2>(tmp1, beta, gamma);
            CHECK(tmp2 == unprimed);
        }

        primed = lorentz_boost<3>(unprimed, gamma * beta);
        CHECK(primed == FourVector(0.09999999999999987, { 0.3, -1.4, 0.7 }));
        CHECK(unprimed == lorentz_boost<3>(primed, -gamma * beta));
        primed = lorentz_boost<-3>(unprimed, -gamma * beta);
        CHECK(primed == FourVector(0.09999999999999987, { 0.3, -1.4, 0.7 }));
        CHECK(unprimed == lorentz_boost<-3>(primed, gamma * beta));
        {
            auto const tmp1 = lorentz_boost<3>(unprimed, beta, gamma);
            CHECK(tmp1 == primed);
            auto const tmp2 = lorentz_boost<-3>(tmp1, beta, gamma);
            CHECK(tmp2 == unprimed);
        }

        primed = lorentz_boost<1>(unprimed, gamma * beta * normal);
        CHECK(primed == FourVector(-0.3488455690265904, { 0.13608466483156534, -1.072169329663131, 1.2180423324157827 }));
        CHECK(unprimed == lorentz_boost<1>(primed, -gamma * beta * normal));
        primed = lorentz_boost<-1>(unprimed, -gamma * beta * normal);
        CHECK(primed == FourVector(-0.3488455690265904, { 0.13608466483156534, -1.072169329663131, 1.2180423324157827 }));
        CHECK(unprimed == lorentz_boost<-1>(primed, gamma * beta * normal));
        {
            auto const tmp1 = lorentz_boost<1>(unprimed, beta * normal, gamma);
            CHECK(tmp1 == primed);
            auto const tmp2 = lorentz_boost<-1>(tmp1, beta * normal, gamma);
            CHECK(tmp2 == unprimed);
        }
    }
}

TEST_CASE("Test LibPIC::SpecialFourVectors", "[LibPIC::SpecialFourVectors]")
{
    constexpr auto vector = FourVector{ 1, { 2, 3, 4 } };
    {
        constexpr auto cart_V = FourCartVector{ vector };
        CHECK(vector == cart_V);

        constexpr auto mfa_V = FourMFAVector{ vector };
        CHECK(vector == mfa_V);

        constexpr auto covar_V = FourCovarVector{ vector };
        CHECK(vector == covar_V);

        constexpr auto contr_V = FourContrVector{ vector };
        CHECK(vector == contr_V);
    }
    {
        constexpr auto scale_factors = FourVector{ 1, { .3, 3, .9384 } };
        constexpr auto covar_V       = FourCovarVector{ vector * scale_factors };
        constexpr auto contr_V       = FourContrVector{ vector / scale_factors };
        CHECK(dot(covar_V, contr_V) == *(vector.t * vector.t) + dot(vector.s, vector.s));
        CHECK(dot(contr_V, covar_V) == *(vector.t * vector.t) + dot(vector.s, vector.s));
    }
    {
        auto const ctau    = 2.0;
        auto const boosted = lorentz_boost<+1>(FourVector{ ctau, {} }, -8.4837 * Vector{ .38, .98, .83 });
        auto const covar   = FourCovarVector{ boosted.t, CovarVector{ +boosted.s } };
        auto const contr   = FourContrVector{ boosted.t, ContrVector{ -boosted.s } };
        CHECK(dot(covar, contr) == Approx{ ctau * ctau }.epsilon(1e-13));
    }
}
