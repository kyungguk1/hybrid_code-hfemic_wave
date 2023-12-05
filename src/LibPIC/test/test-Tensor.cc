/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/VT/FourTensor.h>
#include <PIC/VT/Tensor.h>

namespace {
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
template <class T1, class T2, class T3, class U1, class U2, class U3>
[[nodiscard]] bool operator==(Detail::FourTensorTemplate<T1, T2, T3> const &a, Detail::FourTensorTemplate<U1, U2, U3> const &b) noexcept
{
    return a.tt == Approx{ b.tt }.margin(1e-15)
        && a.ts == b.ts
        && a.ss == b.ss;
}
} // namespace
using ::operator==;
using std::get;

TEST_CASE("Test LibPIC::Tensor", "[LibPIC::Tensor]")
{
    {
        constexpr Tensor t1{};
        constexpr bool   tf = t1.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
          });
        CHECK(tf);

        constexpr Tensor t2{ 1 };
        CHECK(t2.fold(true, [](bool lhs, auto rhs) {
            return lhs && rhs == 1;
        }));

        constexpr Tensor t3 = [](Tensor vv) {
            get<0, 0>(vv) = 1;
            get<1, 1>(vv) = 2;
            get<2, 2>(vv) = 3;
            get<0, 1>(vv) = 4;
            get<1, 2>(vv) = 5;
            get<2, 0>(vv) = 6;
            return vv;
        }({});
        static_assert(get<0, 0>(t3) == 1);
        static_assert(get<1, 1>(t3) == 2);
        static_assert(get<2, 2>(t3) == 3);
        static_assert(get<0, 1>(t3) == 4);
        static_assert(get<1, 2>(t3) == 5);
        static_assert(get<2, 0>(t3) == 6);
        CHECK((t3.xx == 1 && t3.yy == 2 && t3.zz == 3));
        CHECK((t3.xy == 4 && t3.yz == 5 && t3.zx == 6));
        constexpr auto tmp = Tensor{ 1, 2, 3, 4, 5, 6 };
        CHECK(t3 == tmp);

        constexpr Tensor asym = [](Tensor vv) {
            get<1, 0>(vv) = 4;
            get<2, 1>(vv) = 5;
            get<0, 2>(vv) = 6;
            return vv;
        }({});
        static_assert(get<1, 0>(asym) == 4);
        static_assert(get<2, 1>(asym) == 5);
        static_assert(get<0, 2>(asym) == 6);
        CHECK((asym.xy == 4 && asym.yz == 5 && asym.zx == 6));

        constexpr bool tf2 = std::addressof(t1) == std::addressof(+t1);
        CHECK(tf2);

        constexpr Tensor t4 = -t3;
        CHECK((t4.xx == -1 && t4.yy == -2 && t4.zz == -3));
        CHECK((t4.xy == -4 && t4.yz == -5 && t4.zx == -6));

        constexpr Vector v1{ 2, 4, 5 };
        constexpr auto   dot1 = dot(v1, t3);
        CHECK(dot1 == Vector{ 48, 41, 47 });
        constexpr auto dot2 = dot(t3, v1);
        CHECK(dot2 == Vector{ 48, 41, 47 });

        CHECK(t3.lo() == Vector{ 1, 2, 3 });
        CHECK(t3.hi() == Vector{ 4, 5, 6 });

        CHECK(Tensor::identity() == Tensor{ 1, 1, 1, 0, 0, 0 });

        constexpr auto tr = trace(t3);
        CHECK(tr == Approx{ t3.lo().fold(double{}, std::plus{}) }.epsilon(1e-15));

        CHECK(det(t3) == Approx{ 101 }.epsilon(1e-10));

        constexpr auto Tinv_ref = Tensor{ -19, -33, -14, 18, 19, 8 } / 101;
        constexpr auto Tinv     = inv(t3);
        CHECK(Tinv == Tinv_ref);
    }

    {
        constexpr auto is_equal = [](Tensor lhs, Tensor rhs) {
            return lhs.xx == rhs.xx && lhs.yy == rhs.yy
                && lhs.zz == rhs.zz && lhs.xy == rhs.xy
                && lhs.yz == rhs.yz && lhs.zx == rhs.zx;
        };

        constexpr Tensor v1{ 1, 2, 3, -1, -2, -3 };
        CHECK(is_equal(v1, transpose(v1)));

        constexpr double x{ 1 };
        CHECK(is_equal(v1 + x, { 2, 3, 4, -0, -1, -2 }));
        CHECK(is_equal(v1 - x, { 0, 1, 2, -2, -3, -4 }));
        CHECK(is_equal(v1 * x, v1));
        CHECK(is_equal(v1 / x, v1));
        CHECK(is_equal(x + v1, v1 + x));
        CHECK(is_equal(x - v1, -(v1 - x)));
        CHECK(is_equal(x * v1, v1));
        CHECK(is_equal(x / v1, { x / 1, x / 2, x / 3, -x / 1, -x / 2, -x / 3 }));

        constexpr Tensor v2 = v1 * 10.;
        CHECK(is_equal(v1 + v2, { 11, 22, 33, -11, -22, -33 }));
        CHECK(is_equal(v2 - v1, { 9, 18, 27, -9, -18, -27 }));
        CHECK(is_equal(v1 * v2, { 10, 40, 90, 10, 40, 90 }));
        CHECK(is_equal(v2 / v1, { 10, 10, 10, 10, 10, 10 }));
    }
}

TEST_CASE("Test LibPIC::FourTensor", "[LibPIC::FourTensor]")
{
    {
        constexpr FourTensor t1{};
        CHECK(*t1.tt == 0);
        CHECK((t1.ts.x == 0 && t1.ts.y == 0 && t1.ts.z == 0));
        CHECK((t1.ss.xx == 0 && t1.ss.yy == 0 && t1.ss.zz == 0));
        CHECK((t1.ss.xy == 0 && t1.ss.yz == 0 && t1.ss.zx == 0));

        constexpr FourTensor t2{ 1 };
        CHECK(*t2.tt == 1);
        CHECK((t2.ts.x == 1 && t2.ts.y == 1 && t2.ts.z == 1));
        CHECK((t2.ss.xx == 1 && t2.ss.yy == 1 && t2.ss.zz == 1));
        CHECK((t2.ss.xy == 1 && t2.ss.yz == 1 && t2.ss.zx == 1));

        constexpr FourTensor t3 = [](FourTensor ft) {
            get<0, 0>(ft) = -1;
            get<0, 1>(ft) = 10;
            get<0, 2>(ft) = 20;
            get<0, 3>(ft) = 30;
            get<1, 1>(ft) = 1;
            get<2, 2>(ft) = 2;
            get<3, 3>(ft) = 3;
            get<1, 2>(ft) = 4;
            get<2, 3>(ft) = 5;
            get<3, 1>(ft) = 6;
            return ft;
        }({});
        static_assert(*get<0, 0>(t3) == -1);
        static_assert(get<0, 1>(t3) == 10);
        static_assert(get<0, 2>(t3) == 20);
        static_assert(get<0, 3>(t3) == 30);
        static_assert(get<1, 1>(t3) == 1);
        static_assert(get<2, 2>(t3) == 2);
        static_assert(get<3, 3>(t3) == 3);
        static_assert(get<1, 2>(t3) == 4);
        static_assert(get<2, 3>(t3) == 5);
        static_assert(get<3, 1>(t3) == 6);
        CHECK(*t3.tt == -1);
        CHECK((t3.ts.x == 10 && t3.ts.y == 20 && t3.ts.z == 30));
        CHECK((t3.ss.xx == 1 && t3.ss.yy == 2 && t3.ss.zz == 3));
        CHECK((t3.ss.xy == 4 && t3.ss.yz == 5 && t3.ss.zx == 6));
        constexpr auto tmp = FourTensor{ -1, { 10, 20, 30 }, { 1, 2, 3, 4, 5, 6 } };
        CHECK(t3 == tmp);

        constexpr FourTensor asym = [](FourTensor ft) {
            get<1, 0>(ft) = 10;
            get<2, 0>(ft) = 20;
            get<3, 0>(ft) = 30;
            get<2, 1>(ft) = 4;
            get<3, 2>(ft) = 5;
            get<1, 3>(ft) = 6;
            return ft;
        }({});
        static_assert(get<1, 0>(asym) == 10);
        static_assert(get<2, 0>(asym) == 20);
        static_assert(get<3, 0>(asym) == 30);
        static_assert(get<2, 1>(asym) == 4);
        static_assert(get<3, 2>(asym) == 5);
        static_assert(get<1, 3>(asym) == 6);
        CHECK((asym.ts.x == 10 && asym.ts.y == 20 && asym.ts.z == 30));
        CHECK((asym.ss.xy == 4 && asym.ss.yz == 5 && asym.ss.zx == 6));

        constexpr bool tf2 = std::addressof(t1) == std::addressof(+t1);
        CHECK(tf2);

        constexpr FourTensor t4 = -t3;
        CHECK(*t4.tt == 1);
        CHECK((t4.ts.x == -10 && t4.ts.y == -20 && t4.ts.z == -30));
        CHECK((t4.ss.xx == -1 && t4.ss.yy == -2 && t4.ss.zz == -3));
        CHECK((t4.ss.xy == -4 && t4.ss.yz == -5 && t4.ss.zx == -6));

        constexpr auto minkowski_metric = FourTensor::minkowski_metric();
        static_assert(*get<0, 0>(minkowski_metric) == 1);
        static_assert(get<0, 1>(minkowski_metric) == 0);
        static_assert(get<0, 2>(minkowski_metric) == 0);
        static_assert(get<0, 3>(minkowski_metric) == 0);
        static_assert(get<1, 1>(minkowski_metric) == -1);
        static_assert(get<2, 2>(minkowski_metric) == -1);
        static_assert(get<3, 3>(minkowski_metric) == -1);
        static_assert(get<1, 2>(minkowski_metric) == 0);
        static_assert(get<2, 3>(minkowski_metric) == 0);
        static_assert(get<3, 1>(minkowski_metric) == 0);
    }

    {
        constexpr auto is_equal = [](FourTensor lhs, FourTensor rhs) {
            return *lhs.tt == *rhs.tt
                && lhs.ts.x == rhs.ts.x && lhs.ts.y == rhs.ts.y && lhs.ts.z == rhs.ts.z
                && lhs.ss.xx == rhs.ss.xx && lhs.ss.yy == rhs.ss.yy && lhs.ss.zz == rhs.ss.zz
                && lhs.ss.xy == rhs.ss.xy && lhs.ss.yz == rhs.ss.yz && lhs.ss.zx == rhs.ss.zx;
        };

        constexpr FourTensor v1{ -1, { 1, 2, 3 }, { 1, 2, 3, -1, -2, -3 } };
        constexpr double     x{ 1 };
        CHECK(is_equal(v1 + x, { 0, { 2, 3, 4 }, { 2, 3, 4, -0, -1, -2 } }));
        CHECK(is_equal(v1 - x, { -2, { 0, 1, 2 }, { 0, 1, 2, -2, -3, -4 } }));
        CHECK(is_equal(v1 * x, v1));
        CHECK(is_equal(v1 / x, v1));
        CHECK(is_equal(x + v1, v1 + x));
        CHECK(is_equal(x - v1, -(v1 - x)));
        CHECK(is_equal(x * v1, v1));
        CHECK(is_equal(x / v1, { -x / 1, { x / 1, x / 2, x / 3 }, { x / 1, x / 2, x / 3, -x / 1, -x / 2, -x / 3 } }));

        constexpr FourTensor v2 = v1 * 10.;
        CHECK(is_equal(v1 + v2, { -11, { 11, 22, 33 }, { 11, 22, 33, -11, -22, -33 } }));
        CHECK(is_equal(v2 - v1, { -9, { 9, 18, 27 }, { 9, 18, 27, -9, -18, -27 } }));
        CHECK(is_equal(v1 * v2, { 10, { 10, 40, 90 }, { 10, 40, 90, 10, 40, 90 } }));
        CHECK(is_equal(v2 / v1, { 10, { 10, 10, 10 }, { 10, 10, 10, 10, 10, 10 } }));
    }

    { // boost in x dir
        constexpr auto beta  = .8;
        constexpr auto gamma = 1.666666666666667;
        constexpr auto F_lab = FourTensor{
            1.4970372805041228,
            { 0.059690862734654626, 1.7044582124973227, -1.701656846506585 },
            { 0.15549715324872704, -0.9984071812175177, -1.7474254169645187, 0.28329944048187894, -0.5261110361350201, 0.17427786541039625 }
        };
        constexpr auto F_co = FourTensor{
            4.169583550577391,
            { -3.40037370032624, 2.463031100186366, -3.068465231391504 },
            { 2.828043423321995, -0.998407181217518, -1.747425416964519, -1.800445215859966, -0.5261110361350201, 2.559338904359441 }
        };

        auto const primed = lorentz_boost<+1>(F_lab, beta * gamma);
        CHECK(primed == F_co);
        auto const unprimed = lorentz_boost<-1>(primed, beta * gamma);
        CHECK(unprimed == F_lab);
        auto const tmp1 = lorentz_boost<+1>(F_lab, beta, gamma);
        CHECK(F_co == tmp1);
        auto const tmp2 = lorentz_boost<-1>(F_co, beta, gamma);
        CHECK(F_lab == tmp2);
    }
    { // boost in y dir
        constexpr auto beta  = .8;
        constexpr auto gamma = 1.666666666666667;
        constexpr auto F_lab = FourTensor{
            1.4970372805041228,
            { 0.059690862734654626, 1.7044582124973227, -1.701656846506585 },
            { 0.15549715324872704, -0.9984071812175177, -1.7474254169645187, 0.28329944048187894, -0.5261110361350201, 0.17427786541039625 }
        };
        constexpr auto F_co = FourTensor{
            -5.19187904297446,
            { -0.2782478160847476, 6.656687191850904, -2.134613362664282 },
            { 0.155497153248727, -7.687323504696099, -1.747425416964519, 0.3925779171569253, 1.392024068450414, 0.1742778654103962 }
        };

        auto const primed = lorentz_boost<+2>(F_lab, beta * gamma);
        CHECK(primed == F_co);
        auto const unprimed = lorentz_boost<-2>(primed, beta * gamma);
        CHECK(unprimed == F_lab);
        auto const tmp1 = lorentz_boost<+2>(F_lab, beta, gamma);
        CHECK(F_co == tmp1);
        auto const tmp2 = lorentz_boost<-2>(F_co, beta, gamma);
        CHECK(F_lab == tmp2);
    }
    { // boost in z dir
        constexpr auto beta  = .8;
        constexpr auto gamma = 1.666666666666667;
        constexpr auto F_lab = FourTensor{
            1.4970372805041228,
            { 0.059690862734654626, 1.7044582124973227, -1.701656846506585 },
            { 0.15549715324872704, -0.9984071812175177, -1.7474254169645187, 0.28329944048187894, -0.5261110361350201, 0.17427786541039625 }
        };
        constexpr auto F_co = FourTensor{
            8.61482213349269,
            { -0.1328857159894373, 3.542245069008898, -7.19557421972912 },
            { 0.155497153248727, -0.998407181217518, 5.370359436024046, 0.2832994404818789, -3.149462676888131, 0.2108752920377876 }
        };

        auto const primed = lorentz_boost<+3>(F_lab, beta * gamma);
        CHECK(primed == F_co);
        auto const unprimed = lorentz_boost<-3>(primed, beta * gamma);
        CHECK(unprimed == F_lab);
        auto const tmp1 = lorentz_boost<+3>(F_lab, beta, gamma);
        CHECK(F_co == tmp1);
        auto const tmp2 = lorentz_boost<-3>(F_co, beta, gamma);
        CHECK(F_lab == tmp2);
    }
    { // boost in arbitrary dir
        constexpr auto beta  = .8 * Vector{ -0.6005538266461359, 0.7508361240815488, 0.2749185626233338 };
        constexpr auto gamma = 1.666666666666667;
        constexpr auto F_lab = FourTensor{
            1.4970372805041228,
            { 0.059690862734654626, 1.7044582124973227, -1.701656846506585 },
            { 0.15549715324872704, -0.9984071812175177, -1.7474254169645187, 0.28329944048187894, -0.5261110361350201, 0.17427786541039625 }
        };
        constexpr auto F_co = FourTensor{
            -1.369280913843746,
            { -0.0961830222491265, 4.22582902166098, -1.541818734702444 },
            { 0.3920267375108505, -5.047105841634422, -0.801574535157607, 1.754609338103473, 0.02429074158349565, -0.912956500649025 }
        };

        auto const primed = lorentz_boost<+1>(F_lab, beta * gamma);
        CHECK(primed == F_co);
        auto const unprimed = lorentz_boost<-1>(primed, beta * gamma);
        CHECK(unprimed == F_lab);
        auto const tmp1 = lorentz_boost<+1>(F_lab, beta, gamma);
        CHECK(F_co == tmp1);
        auto const tmp2 = lorentz_boost<-1>(F_co, beta, gamma);
        CHECK(F_lab == tmp2);
    }
}

TEST_CASE("Test LibPIC::SpecialTensors", "[LibPIC::SpecialTensors]")
{
    {
        constexpr auto tensor = Tensor{ 1, 2, 3, 4, 5, 6 };

        constexpr auto cart = CartTensor{ tensor };
        CHECK(tensor == cart);

        constexpr auto mfa = MFATensor{ tensor };
        CHECK(tensor == mfa);

        constexpr auto covar = CovarTensor{ tensor };
        CHECK(tensor == covar);

        constexpr auto contr = ContrTensor{ tensor };
        CHECK(tensor == contr);
    }
    {
        constexpr auto scale_factors = Vector{ .3, 3, .9384 };
        constexpr auto vector        = Vector{ .59, -.9483, .4958 };
        constexpr auto covarV        = CovarVector{ vector * scale_factors };
        constexpr auto covarT        = CovarTensor{
            covarV.x * covarV.x,
            covarV.y * covarV.y,
            covarV.z * covarV.z,
            covarV.x * covarV.y,
            covarV.y * covarV.z,
            covarV.z * covarV.x
        };
        constexpr auto contrV = ContrVector{ vector / scale_factors };
        constexpr auto contrT = ContrTensor{
            contrV.x * contrV.x,
            contrV.y * contrV.y,
            contrV.z * contrV.z,
            contrV.x * contrV.y,
            contrV.y * contrV.z,
            contrV.z * contrV.x
        };
        CHECK(dot(contrV, dot(covarT, contrV)) == Approx{ dot(covarV, dot(contrT, covarV)) }.epsilon(1e-15));
        CHECK(dot(dot(contrV, covarT), contrV) == Approx{ dot(dot(covarV, contrT), covarV) }.epsilon(1e-15));
    }
}

TEST_CASE("Test LibPIC::SpecialFourTensors", "[LibPIC::SpecialFourTensors]")
{
    {
        constexpr auto tensor = FourTensor{ .5, { -1, -2, -3 }, { 1, 2, 3, 4, 5, 6 } };

        constexpr auto cart = FourCartTensor{ tensor };
        CHECK(tensor == cart);

        constexpr auto mfa = FourMFATensor{ tensor };
        CHECK(tensor == mfa);

        constexpr auto covar = FourCovarTensor{ tensor };
        CHECK(tensor == covar);

        constexpr auto contr = FourContrTensor{ tensor };
        CHECK(tensor == contr);
    }
    {
        auto const ctau    = 2.0;
        auto const boosted = lorentz_boost<+1>(FourVector{ ctau, {} }, -8.4837 * Vector{ .38, .98, .83 });
        auto const covarV  = FourCovarVector{ boosted.t, CovarVector{ +boosted.s } };
        auto const covarT  = FourCovarTensor{
            covarV.t * covarV.t,
            *covarV.t * covarV.s,
            {
                covarV.s.x * covarV.s.x,
                covarV.s.y * covarV.s.y,
                covarV.s.z * covarV.s.z,
                covarV.s.x * covarV.s.y,
                covarV.s.y * covarV.s.z,
                covarV.s.z * covarV.s.x,
            }
        };
        auto const contrV = FourContrVector{ boosted.t, ContrVector{ -boosted.s } };
        auto const contrT = FourContrTensor{
            contrV.t * contrV.t,
            *contrV.t * contrV.s,
            {
                contrV.s.x * contrV.s.x,
                contrV.s.y * contrV.s.y,
                contrV.s.z * contrV.s.z,
                contrV.s.x * contrV.s.y,
                contrV.s.y * contrV.s.z,
                contrV.s.z * contrV.s.x,
            }
        };
        CHECK(dot(dot(covarT, contrV), contrV) == Approx{ dot(dot(contrT, covarV), covarV) }.epsilon(1e-15));
        CHECK(dot(dot(covarT, contrV), contrV) == Approx{ std::pow(ctau, 4) }.epsilon(1e-10));
        CHECK(dot(contrV, dot(contrV, covarT)) == Approx{ dot(covarV, dot(covarV, contrT)) }.epsilon(1e-15));
        CHECK(dot(contrV, dot(contrV, covarT)) == Approx{ std::pow(ctau, 4) }.epsilon(1e-10));
    }
}
