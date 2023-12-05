/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/Geometry.h>

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
} // namespace
using ::operator==;

TEST_CASE("Test LibPIC::Geometry::Ctor", "[LibPIC::Geometry::Ctor]")
{
    CHECK_THROWS_AS(Geometry(-1, 1, 1), std::invalid_argument);
    CHECK_THROWS_AS(Geometry(1, 0, 1), std::invalid_argument);
    CHECK_THROWS_AS(Geometry(0, 1, -1), std::invalid_argument);
    CHECK_NOTHROW(Geometry(0, 1, 1));

    constexpr auto O0 = 1;

    { // homogeneous
        constexpr Real xi = 1e-11;
        constexpr Real D1 = 0.1;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        CHECK(xi == mirror.xi());
        CHECK(mirror.is_homogeneous());

        CHECK(mirror.D().x == D1);
        CHECK(mirror.D().y == D2);
        CHECK(mirror.D().z == D3);
        CHECK(mirror.D1() == D1);
        CHECK(mirror.D2() == D2);
        CHECK(mirror.D3() == D3);

        CHECK(mirror.sqrt_g() == D1 * D2 * D3);
        CHECK(mirror.det_gij() == mirror.sqrt_g() * mirror.sqrt_g());

        CHECK(mirror.is_valid(CurviCoord{ 0 }));
        CHECK(mirror.is_valid(CurviCoord{ 1 }));
        CHECK(mirror.is_valid(CurviCoord{ -100 }));
    }

    { // inhomogeneous
        constexpr Real xi = 0.112;
        constexpr Real D1 = 2;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        CHECK(xi == mirror.xi());
        CHECK(!mirror.is_homogeneous());

        CHECK(mirror.D().x == D1);
        CHECK(mirror.D().y == D2);
        CHECK(mirror.D().z == D3);
        CHECK(mirror.D1() == D1);
        CHECK(mirror.D2() == D2);
        CHECK(mirror.D3() == D3);

        CHECK(mirror.sqrt_g() == D1 * D2 * D3);
        CHECK(mirror.det_gij() == mirror.sqrt_g() * mirror.sqrt_g());

        CHECK(mirror.is_valid(CurviCoord{ 0 }));
        CHECK(mirror.is_valid(CurviCoord{ M_PI_2 * 0.99999999 / (xi * D1) }));
        CHECK(!mirror.is_valid(CurviCoord{ -M_PI_2 * 1.00000001 / (xi * D1) }));
    }
}

TEST_CASE("Test LibPIC::Geometry::Cotrans", "[LibPIC::Geometry::Cotrans]")
{
    constexpr auto O0 = 1;

    { // homogeneous
        constexpr Real xi = 1e-11;
        constexpr Real D1 = 0.1;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart1{ 14.5 };
        auto const          curvi = mirror.cotrans(cart1);
        CHECK(curvi.q1 * D1 == Approx{ cart1.x }.epsilon(1e-10));
        auto const cart2 = mirror.cotrans(curvi);
        CHECK(cart2.x == Approx{ cart1.x }.epsilon(1e-10));
    }

    { // inhomogeneous
        constexpr Real xi = 0.112;
        constexpr Real D1 = 2;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart1{ 14.5 };
        auto const          curvi = mirror.cotrans(cart1);
        CHECK(curvi.q1 * D1 == Approx{ 9.09702270985558 }.epsilon(1e-10));
        auto const cart2 = mirror.cotrans(curvi);
        CHECK(cart2.x == Approx{ cart1.x }.epsilon(1e-10));
    }
}

TEST_CASE("Test LibPIC::Geometry::Field", "[LibPIC::Geometry::Field]")
{
    constexpr auto O0 = 1;

    { // homogeneous
        constexpr Real xi = 1e-11;
        constexpr Real D1 = 0.1;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 14.5 };
        auto const          curvi = mirror.cotrans(cart);

        CartVector Bcart = mirror.Bcart_div_B0(cart);
        CHECK(Bcart.x == Approx{ 1 }.epsilon(1e-15));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        Bcart = mirror.Bcart_div_B0(curvi);
        CHECK(Bcart.x == Approx{ 1 }.epsilon(1e-15));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        CHECK(std::pow(mirror.Bmag_div_B0(cart), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));
        CHECK(std::pow(mirror.Bmag_div_B0(curvi), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));

        Bcart = mirror.Bcart_div_B0(cart, 10, 20);
        CHECK(Bcart.x == Approx{ 1 }.epsilon(1e-15));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        Bcart = mirror.Bcart_div_B0(curvi, 10, 20);
        CHECK(Bcart.x == Approx{ 1 }.epsilon(1e-15));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        ContrVector Bcontr = mirror.Bcontr_div_B0(cart);
        CHECK(Bcontr.x * D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));

        Bcontr = mirror.Bcontr_div_B0(curvi);
        CHECK(Bcontr.x * D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));

        CovarVector Bcovar = mirror.Bcovar_div_B0(cart);
        CHECK(Bcovar.x / D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));

        Bcovar = mirror.Bcovar_div_B0(curvi);
        CHECK(Bcovar.x / D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));
    }

    { // inhomogeneous
        constexpr Real xi = 0.112;
        constexpr Real D1 = 2;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 14.5 };
        auto const          curvi = mirror.cotrans(cart);

        CartVector Bcart = mirror.Bcart_div_B0(cart);
        CHECK(Bcart.x == Approx{ 3.637376 }.epsilon(1e-10));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        Bcart = mirror.Bcart_div_B0(curvi);
        CHECK(Bcart.x == Approx{ 3.637376 }.epsilon(1e-10));
        CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

        CHECK(std::pow(mirror.Bmag_div_B0(cart), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));
        CHECK(std::pow(mirror.Bmag_div_B0(curvi), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));

        Bcart = mirror.Bcart_div_B0(cart, 10, 20);
        CHECK(Bcart.x == Approx{ 3.637376 }.epsilon(1e-10));
        CHECK(Bcart.y == Approx{ -1.8188800000000003 }.epsilon(1e-10));
        CHECK(Bcart.z == Approx{ -3.6377600000000005 }.epsilon(1e-10));

        Bcart = mirror.Bcart_div_B0(curvi, 10, 20);
        CHECK(Bcart.x == Approx{ 3.637376 }.epsilon(1e-10));
        CHECK(Bcart.y == Approx{ -1.8188800000000003 }.epsilon(1e-10));
        CHECK(Bcart.z == Approx{ -3.6377600000000005 }.epsilon(1e-10));

        ContrVector Bcontr = mirror.Bcontr_div_B0(cart);
        CHECK(Bcontr.x * D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));

        Bcontr = mirror.Bcontr_div_B0(curvi);
        CHECK(Bcontr.x * D1 == Approx{ 1 }.epsilon(1e-10));
        CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));

        CovarVector Bcovar = mirror.Bcovar_div_B0(cart);
        Bcart              = mirror.Bcart_div_B0(cart);
        CHECK(Bcovar.x / D1 == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));
        CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));

        Bcovar = mirror.Bcovar_div_B0(curvi);
        CHECK(Bcovar.x / D1 == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));
        CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
        CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));
    }
}

TEST_CASE("Test LibPIC::Geometry::Basis", "[LibPIC::Geometry::Basis]")
{
    constexpr auto O0 = 1;

    { // homogeneous
        constexpr Real xi = 1e-11;
        constexpr Real D1 = 0.1;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 14.5 };
        auto const          curvi = mirror.cotrans(cart);

        // covar metric
        CovarTensor covar_metric = mirror.covar_metric(cart);
        CHECK(covar_metric.xx == Approx{ D1 * D1 }.epsilon(1e-10));
        CHECK(covar_metric.yy == Approx{ D2 * D2 }.epsilon(1e-10));
        CHECK(covar_metric.zz == Approx{ D3 * D3 }.epsilon(1e-10));
        CHECK(covar_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.zx == Approx{ 0 }.margin(1e-15));

        covar_metric = mirror.covar_metric(curvi);
        CHECK(covar_metric.xx == Approx{ D1 * D1 }.epsilon(1e-10));
        CHECK(covar_metric.yy == Approx{ D2 * D2 }.epsilon(1e-10));
        CHECK(covar_metric.zz == Approx{ D3 * D3 }.epsilon(1e-10));
        CHECK(covar_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.zx == Approx{ 0 }.margin(1e-15));

        // contr metric
        ContrTensor contr_metric = mirror.contr_metric(cart);
        CHECK(1 / contr_metric.xx == Approx{ D1 * D1 }.epsilon(1e-10));
        CHECK(1 / contr_metric.yy == Approx{ D2 * D2 }.epsilon(1e-10));
        CHECK(1 / contr_metric.zz == Approx{ D3 * D3 }.epsilon(1e-10));
        CHECK(contr_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.zx == Approx{ 0 }.margin(1e-15));

        contr_metric = mirror.contr_metric(curvi);
        CHECK(1 / contr_metric.xx == Approx{ D1 * D1 }.epsilon(1e-10));
        CHECK(1 / contr_metric.yy == Approx{ D2 * D2 }.epsilon(1e-10));
        CHECK(1 / contr_metric.zz == Approx{ D3 * D3 }.epsilon(1e-10));
        CHECK(contr_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.zx == Approx{ 0 }.margin(1e-15));

        // covar basis
        CartVector covar_basis = mirror.covar_basis<1>(cart);
        CHECK(covar_basis.x == Approx{ D1 }.epsilon(1e-10));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<2>(cart);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ D2 }.epsilon(1e-10));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<3>(cart);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ D3 }.epsilon(1e-10));
        CartTensor covar_bases = mirror.covar_basis<0>(cart);
        CHECK(covar_bases.xx == Approx{ D1 }.epsilon(1e-10));
        CHECK(covar_bases.yy == Approx{ D2 }.epsilon(1e-10));
        CHECK(covar_bases.zz == Approx{ D3 }.epsilon(1e-10));
        CHECK(covar_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.zx == Approx{ 0 }.margin(1e-15));

        covar_basis = mirror.covar_basis<1>(curvi);
        CHECK(covar_basis.x == Approx{ D1 }.epsilon(1e-10));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<2>(curvi);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ D2 }.epsilon(1e-10));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<3>(curvi);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ D3 }.epsilon(1e-10));
        covar_bases = mirror.covar_basis<0>(curvi);
        CHECK(covar_bases.xx == Approx{ D1 }.epsilon(1e-10));
        CHECK(covar_bases.yy == Approx{ D2 }.epsilon(1e-10));
        CHECK(covar_bases.zz == Approx{ D3 }.epsilon(1e-10));
        CHECK(covar_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.zx == Approx{ 0 }.margin(1e-15));

        // contr basis
        CartVector contr_basis = mirror.contr_basis<1>(cart);
        CHECK(contr_basis.x == Approx{ 1 / D1 }.epsilon(1e-10));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<2>(cart);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 1 / D2 }.epsilon(1e-10));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<3>(cart);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 1 / D3 }.epsilon(1e-10));
        CartTensor contr_bases = mirror.contr_basis<0>(cart);
        CHECK(contr_bases.xx == Approx{ 1 / D1 }.epsilon(1e-10));
        CHECK(contr_bases.yy == Approx{ 1 / D2 }.epsilon(1e-10));
        CHECK(contr_bases.zz == Approx{ 1 / D3 }.epsilon(1e-10));
        CHECK(contr_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.zx == Approx{ 0 }.margin(1e-15));

        contr_basis = mirror.contr_basis<1>(curvi);
        CHECK(contr_basis.x == Approx{ 1 / D1 }.epsilon(1e-10));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<2>(curvi);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 1 / D2 }.epsilon(1e-10));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<3>(curvi);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 1 / D3 }.epsilon(1e-10));
        contr_bases = mirror.contr_basis<0>(curvi);
        CHECK(contr_bases.xx == Approx{ 1 / D1 }.epsilon(1e-10));
        CHECK(contr_bases.yy == Approx{ 1 / D2 }.epsilon(1e-10));
        CHECK(contr_bases.zz == Approx{ 1 / D3 }.epsilon(1e-10));
        CHECK(contr_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.zx == Approx{ 0 }.margin(1e-15));
    }

    { // inhomogeneous
        constexpr Real xi = 0.512;
        constexpr Real D1 = 2;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 7.5121 };
        auto const          curvi = mirror.cotrans(cart);

        // covar metric
        CovarTensor covar_metric = mirror.covar_metric(cart);
        CHECK(covar_metric.xx == Approx{ 997.702878094314 }.epsilon(1e-10));
        CHECK(covar_metric.yy == Approx{ 0.011707557361683246 }.epsilon(1e-10));
        CHECK(covar_metric.zz == Approx{ 0.15016572763097885 }.epsilon(1e-10));
        CHECK(covar_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.zx == Approx{ 0 }.margin(1e-15));

        covar_metric = mirror.covar_metric(curvi);
        CHECK(covar_metric.xx == Approx{ 997.702878094314 }.epsilon(1e-10));
        CHECK(covar_metric.yy == Approx{ 0.011707557361683246 }.epsilon(1e-10));
        CHECK(covar_metric.zz == Approx{ 0.15016572763097885 }.epsilon(1e-10));
        CHECK(covar_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_metric.zx == Approx{ 0 }.margin(1e-15));

        // contr metric
        ContrTensor contr_metric = mirror.contr_metric(cart);
        CHECK(contr_metric.xx == Approx{ 0.0010023024108240257 }.epsilon(1e-10));
        CHECK(contr_metric.yy == Approx{ 85.41491355599265 }.epsilon(1e-10));
        CHECK(contr_metric.zz == Approx{ 6.659309123167078 }.epsilon(1e-10));
        CHECK(contr_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.zx == Approx{ 0 }.margin(1e-15));

        contr_metric = mirror.contr_metric(curvi);
        CHECK(contr_metric.xx == Approx{ 0.0010023024108240257 }.epsilon(1e-10));
        CHECK(contr_metric.yy == Approx{ 85.41491355599265 }.epsilon(1e-10));
        CHECK(contr_metric.zz == Approx{ 6.659309123167078 }.epsilon(1e-10));
        CHECK(contr_metric.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_metric.zx == Approx{ 0 }.margin(1e-15));

        // covar basis
        CartVector covar_basis = mirror.covar_basis<1>(cart);
        CHECK(covar_basis.x == Approx{ 31.586435033006083 }.epsilon(1e-10));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<2>(cart);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0.10820146654127774 }.epsilon(1e-10));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<3>(cart);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0.38751222900829707 }.epsilon(1e-10));
        CartTensor covar_bases = mirror.covar_basis<0>(cart);
        CHECK(covar_bases.xx == Approx{ 31.586435033006083 }.epsilon(1e-10));
        CHECK(covar_bases.yy == Approx{ 0.10820146654127774 }.epsilon(1e-10));
        CHECK(covar_bases.zz == Approx{ 0.38751222900829707 }.epsilon(1e-10));
        CHECK(covar_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.zx == Approx{ 0 }.margin(1e-15));

        covar_basis = mirror.covar_basis<1>(curvi);
        CHECK(covar_basis.x == Approx{ 31.586435033006083 }.epsilon(1e-10));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<2>(curvi);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0.10820146654127774 }.epsilon(1e-10));
        CHECK(covar_basis.z == Approx{ 0 }.margin(1e-15));
        covar_basis = mirror.covar_basis<3>(curvi);
        CHECK(covar_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(covar_basis.z == Approx{ 0.38751222900829707 }.epsilon(1e-10));
        covar_bases = mirror.covar_basis<0>(curvi);
        CHECK(covar_bases.xx == Approx{ 31.586435033006083 }.epsilon(1e-10));
        CHECK(covar_bases.yy == Approx{ 0.10820146654127774 }.epsilon(1e-10));
        CHECK(covar_bases.zz == Approx{ 0.38751222900829707 }.epsilon(1e-10));
        CHECK(covar_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(covar_bases.zx == Approx{ 0 }.margin(1e-15));

        // contr basis
        CartVector contr_basis = mirror.contr_basis<1>(cart);
        CHECK(contr_basis.x == Approx{ 0.03165915998291846 }.epsilon(1e-10));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<2>(cart);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 9.24201891125487 }.epsilon(1e-10));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<3>(cart);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 2.5805637219737627 }.epsilon(1e-10));
        CartTensor contr_bases = mirror.contr_basis<0>(cart);
        CHECK(contr_bases.xx == Approx{ 0.03165915998291846 }.epsilon(1e-10));
        CHECK(contr_bases.yy == Approx{ 9.24201891125487 }.epsilon(1e-10));
        CHECK(contr_bases.zz == Approx{ 2.5805637219737627 }.epsilon(1e-10));
        CHECK(contr_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.zx == Approx{ 0 }.margin(1e-15));

        contr_basis = mirror.contr_basis<1>(curvi);
        CHECK(contr_basis.x == Approx{ 0.03165915998291846 }.epsilon(1e-10));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<2>(curvi);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 9.24201891125487 }.epsilon(1e-10));
        CHECK(contr_basis.z == Approx{ 0 }.margin(1e-15));
        contr_basis = mirror.contr_basis<3>(curvi);
        CHECK(contr_basis.x == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.y == Approx{ 0 }.margin(1e-15));
        CHECK(contr_basis.z == Approx{ 2.5805637219737627 }.epsilon(1e-10));
        contr_bases = mirror.contr_basis<0>(curvi);
        CHECK(contr_bases.xx == Approx{ 0.03165915998291846 }.epsilon(1e-10));
        CHECK(contr_bases.yy == Approx{ 9.24201891125487 }.epsilon(1e-10));
        CHECK(contr_bases.zz == Approx{ 2.5805637219737627 }.epsilon(1e-10));
        CHECK(contr_bases.xy == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.yz == Approx{ 0 }.margin(1e-15));
        CHECK(contr_bases.zx == Approx{ 0 }.margin(1e-15));
    }
}

TEST_CASE("Test LibPIC::Geometry::Transform", "[LibPIC::Geometry::Transform]")
{
    constexpr auto O0 = 1;

    { // homogeneous
        constexpr Real xi = 1e-11;
        constexpr Real D1 = 0.1;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 14.5 };
        auto const          curvi = mirror.cotrans(cart);
        CartVector const    vcart{ 1.3, .506, -.598 };
        CovarVector const   vcovar{ vcart.x * D1, vcart.y * D2, vcart.z * D3 };
        ContrVector const   vcontr{ vcart.x / D1, vcart.y / D2, vcart.z / D3 };

        CHECK(mirror.contr_to_covar(vcontr, cart) == vcovar);
        CHECK(mirror.contr_to_covar(vcontr, curvi) == vcovar);

        CHECK(mirror.covar_to_contr(vcovar, cart) == vcontr);
        CHECK(mirror.covar_to_contr(vcovar, curvi) == vcontr);

        CHECK(mirror.cart_to_contr(vcart, cart) == vcontr);
        CHECK(mirror.cart_to_contr(vcart, curvi) == vcontr);

        CHECK(mirror.contr_to_cart(vcontr, cart) == vcart);
        CHECK(mirror.contr_to_cart(vcontr, curvi) == vcart);

        CHECK(mirror.cart_to_covar(vcart, cart) == vcovar);
        CHECK(mirror.cart_to_covar(vcart, curvi) == vcovar);

        CHECK(mirror.covar_to_cart(vcovar, cart) == vcart);
        CHECK(mirror.covar_to_cart(vcovar, curvi) == vcart);
    }

    { // inhomogeneous
        constexpr Real xi = 0.512;
        constexpr Real D1 = 2;
        constexpr Real D2 = 0.43;
        constexpr Real D3 = 1.54;
        Geometry const mirror{ xi, { D1, D2, D3 }, O0 };

        constexpr CartCoord cart{ 7.5121 };
        auto const          curvi = mirror.cotrans(cart);
        constexpr Vector    vec{ 1.3, .506, -.598 };

        CovarVector tmp_covar = mirror.contr_to_covar(ContrVector{ vec }, cart);
        CHECK(tmp_covar.x == Approx{ 1297.0137415226081 }.epsilon(1e-10));
        CHECK(tmp_covar.y == Approx{ 0.005924024025011723 }.epsilon(1e-10));
        CHECK(tmp_covar.z == Approx{ -0.08979910512332535 }.epsilon(1e-10));
        tmp_covar = mirror.contr_to_covar(ContrVector{ vec }, curvi);
        CHECK(tmp_covar.x == Approx{ 1297.0137415226081 }.epsilon(1e-10));
        CHECK(tmp_covar.y == Approx{ 0.005924024025011723 }.epsilon(1e-10));
        CHECK(tmp_covar.z == Approx{ -0.08979910512332535 }.epsilon(1e-10));

        ContrVector tmp_contr = mirror.covar_to_contr(CovarVector{ vec }, cart);
        CHECK(tmp_contr.x == Approx{ 0.0013029931340712334 }.epsilon(1e-10));
        CHECK(tmp_contr.y == Approx{ 43.21994625933228 }.epsilon(1e-10));
        CHECK(tmp_contr.z == Approx{ -3.9822668556539123 }.epsilon(1e-10));
        tmp_contr = mirror.covar_to_contr(CovarVector{ vec }, curvi);
        CHECK(tmp_contr.x == Approx{ 0.0013029931340712334 }.epsilon(1e-10));
        CHECK(tmp_contr.y == Approx{ 43.21994625933228 }.epsilon(1e-10));
        CHECK(tmp_contr.z == Approx{ -3.9822668556539123 }.epsilon(1e-10));

        tmp_contr = mirror.cart_to_contr(CartVector{ vec }, cart);
        CHECK(tmp_contr.x == Approx{ 0.041156907977794 }.epsilon(1e-10));
        CHECK(tmp_contr.y == Approx{ 4.676461569094965 }.epsilon(1e-10));
        CHECK(tmp_contr.z == Approx{ -1.54317710574031 }.epsilon(1e-10));
        tmp_contr = mirror.cart_to_contr(CartVector{ vec }, curvi);
        CHECK(tmp_contr.x == Approx{ 0.041156907977794 }.epsilon(1e-10));
        CHECK(tmp_contr.y == Approx{ 4.676461569094965 }.epsilon(1e-10));
        CHECK(tmp_contr.z == Approx{ -1.54317710574031 }.epsilon(1e-10));

        CartVector tmp_cart = mirror.contr_to_cart(ContrVector{ vec }, cart);
        CHECK(tmp_cart.x == Approx{ 41.06236554290791 }.epsilon(1e-10));
        CHECK(tmp_cart.y == Approx{ 0.05474994206988654 }.epsilon(1e-10));
        CHECK(tmp_cart.z == Approx{ -0.23173231294696164 }.epsilon(1e-10));
        tmp_cart = mirror.contr_to_cart(ContrVector{ vec }, curvi);
        CHECK(tmp_cart.x == Approx{ 41.06236554290791 }.epsilon(1e-10));
        CHECK(tmp_cart.y == Approx{ 0.05474994206988654 }.epsilon(1e-10));
        CHECK(tmp_cart.z == Approx{ -0.23173231294696164 }.epsilon(1e-10));

        tmp_covar = mirror.cart_to_covar(CartVector{ vec }, cart);
        CHECK(tmp_covar.x == Approx{ 41.06236554290791 }.epsilon(1e-10));
        CHECK(tmp_covar.y == Approx{ 0.05474994206988654 }.epsilon(1e-10));
        CHECK(tmp_covar.z == Approx{ -0.23173231294696164 }.epsilon(1e-10));
        tmp_covar = mirror.cart_to_covar(CartVector{ vec }, curvi);
        CHECK(tmp_covar.x == Approx{ 41.06236554290791 }.epsilon(1e-10));
        CHECK(tmp_covar.y == Approx{ 0.05474994206988654 }.epsilon(1e-10));
        CHECK(tmp_covar.z == Approx{ -0.23173231294696164 }.epsilon(1e-10));

        tmp_cart = mirror.covar_to_cart(CovarVector{ vec }, cart);
        CHECK(tmp_cart.x == Approx{ 0.041156907977794 }.epsilon(1e-10));
        CHECK(tmp_cart.y == Approx{ 4.676461569094965 }.epsilon(1e-10));
        CHECK(tmp_cart.z == Approx{ -1.54317710574031 }.epsilon(1e-10));
        tmp_cart = mirror.covar_to_cart(CovarVector{ vec }, curvi);
        CHECK(tmp_cart.x == Approx{ 0.041156907977794 }.epsilon(1e-10));
        CHECK(tmp_cart.y == Approx{ 4.676461569094965 }.epsilon(1e-10));
        CHECK(tmp_cart.z == Approx{ -1.54317710574031 }.epsilon(1e-10));
    }
}

TEST_CASE("Test LibPIC::Geometry", "[LibPIC::Geometry]")
{
    constexpr Real xi = 0.512;
    constexpr Real D1 = 2;
    constexpr Real B0 = 2.40939;
    Geometry const geo{ xi, D1, B0 };

    CHECK(geo.B0() == B0);

    Vector     v;
    Tensor     t;
    FourVector fv;
    FourTensor ft;
    CHECK(CartVector{ v } == geo.mfa_to_cart(MFAVector{ v }, CartCoord{}));
    CHECK(CartVector{ v } == geo.mfa_to_cart(MFAVector{ v }, CurviCoord{}));
    CHECK(CartTensor{ t } == geo.mfa_to_cart(MFATensor{ t }, CartCoord{}));
    CHECK(CartTensor{ t } == geo.mfa_to_cart(MFATensor{ t }, CurviCoord{}));
    CHECK(FourCartVector{ fv } == geo.mfa_to_cart(FourMFAVector{ fv }, CartCoord{}));
    CHECK(FourCartVector{ fv } == geo.mfa_to_cart(FourMFAVector{ fv }, CurviCoord{}));
    CHECK(FourCartTensor{ ft } == geo.mfa_to_cart(FourMFATensor{ ft }, CartCoord{}));
    CHECK(FourCartTensor{ ft } == geo.mfa_to_cart(FourMFATensor{ ft }, CurviCoord{}));

    CHECK(MFAVector{ v } == geo.cart_to_mfa(CartVector{ v }, CartCoord{}));
    CHECK(MFAVector{ v } == geo.cart_to_mfa(CartVector{ v }, CurviCoord{}));
    CHECK(MFATensor{ t } == geo.cart_to_mfa(CartTensor{ t }, CartCoord{}));
    CHECK(MFATensor{ t } == geo.cart_to_mfa(CartTensor{ t }, CurviCoord{}));
    CHECK(FourMFAVector{ fv } == geo.cart_to_mfa(FourCartVector{ fv }, CartCoord{}));
    CHECK(FourMFAVector{ fv } == geo.cart_to_mfa(FourCartVector{ fv }, CurviCoord{}));
    CHECK(FourMFATensor{ ft } == geo.cart_to_mfa(FourCartTensor{ ft }, CartCoord{}));
    CHECK(FourMFATensor{ ft } == geo.cart_to_mfa(FourCartTensor{ ft }, CurviCoord{}));

    CHECK(geo.e1(CurviCoord{ 0 }).x == Approx{ 1 }.epsilon(1e-15));
    CHECK(geo.e1(CurviCoord{ 0 }).y == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e1(CurviCoord{ 0 }).z == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e2(CurviCoord{ 0 }).x == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e2(CurviCoord{ 0 }).y == Approx{ 1 }.epsilon(1e-15));
    CHECK(geo.e2(CurviCoord{ 0 }).z == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e3(CurviCoord{ 0 }).x == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e3(CurviCoord{ 0 }).y == Approx{ 0 }.margin(1e-15));
    CHECK(geo.e3(CurviCoord{ 0 }).z == Approx{ 1 }.epsilon(1e-15));

    constexpr CartCoord cart{ 4.5121 };
    auto const          curvi = geo.cotrans(cart);

    ContrVector Bcontr = geo.Bcontr(cart);
    CHECK(Bcontr.x / B0 == Approx{ geo.Bcontr_div_B0(curvi).x }.epsilon(1e-10));
    CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));
    Bcontr = geo.Bcontr(curvi);
    CHECK(Bcontr.x / B0 == Approx{ geo.Bcontr_div_B0(cart).x }.epsilon(1e-10));
    CHECK(Bcontr.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcontr.z == Approx{ 0 }.margin(1e-15));

    CovarVector Bcovar = geo.Bcovar(cart);
    CHECK(Bcovar.x / B0 == Approx{ geo.Bcovar_div_B0(curvi).x }.epsilon(1e-10));
    CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));
    Bcovar = geo.Bcovar(curvi);
    CHECK(Bcovar.x / B0 == Approx{ geo.Bcovar_div_B0(cart).x }.epsilon(1e-10));
    CHECK(Bcovar.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcovar.z == Approx{ 0 }.margin(1e-15));

    CartVector Bcart = geo.Bcart(cart);
    CHECK(Bcart.x / B0 == Approx{ geo.Bcart_div_B0(curvi).x }.epsilon(1e-10));
    CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));
    Bcart = geo.Bcart(curvi);
    CHECK(Bcart.x / B0 == Approx{ geo.Bcart_div_B0(cart).x }.epsilon(1e-10));
    CHECK(Bcart.y == Approx{ 0 }.margin(1e-15));
    CHECK(Bcart.z == Approx{ 0 }.margin(1e-15));

    CHECK(std::pow(geo.Bmag(cart), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));
    CHECK(std::pow(geo.Bmag(curvi), 2) == Approx{ dot(Bcart, Bcart) }.epsilon(1e-10));

    Bcart         = geo.Bcart(cart, 3, 2);
    CartVector B2 = geo.Bcart_div_B0(curvi, 3, 2);
    CHECK(Bcart.x / B0 == Approx{ B2.x }.epsilon(1e-10));
    CHECK(Bcart.y / B0 == Approx{ B2.y }.epsilon(1e-10));
    CHECK(Bcart.z / B0 == Approx{ B2.z }.epsilon(1e-10));
    Bcart = geo.Bcart(curvi, 3, 2);
    B2    = geo.Bcart_div_B0(cart, 3, 2);
    CHECK(Bcart.x / B0 == Approx{ B2.x }.epsilon(1e-10));
    CHECK(Bcart.y / B0 == Approx{ B2.y }.epsilon(1e-10));
    CHECK(Bcart.z / B0 == Approx{ B2.z }.epsilon(1e-10));
}
