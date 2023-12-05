/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/VT/Matrix.h>

TEST_CASE("Test LibPIC::Matrix", "[LibPIC::Matrix]")
{
    {
        Matrix        M;
        Matrix const &cM = M;
        CHECK(&M.m11() == &M.x.x);
        CHECK(&M.m12() == &M.x.y);
        CHECK(&M.m13() == &M.x.z);
        CHECK(&M.m21() == &M.y.x);
        CHECK(&M.m22() == &M.y.y);
        CHECK(&M.m23() == &M.y.z);
        CHECK(&M.m31() == &M.z.x);
        CHECK(&M.m32() == &M.z.y);
        CHECK(&M.m33() == &M.z.z);
        CHECK(&cM.m11() == &M.x.x);
        CHECK(&cM.m12() == &M.x.y);
        CHECK(&cM.m13() == &M.x.z);
        CHECK(&cM.m21() == &M.y.x);
        CHECK(&cM.m22() == &M.y.y);
        CHECK(&cM.m23() == &M.y.z);
        CHECK(&cM.m31() == &M.z.x);
        CHECK(&cM.m32() == &M.z.y);
        CHECK(&cM.m33() == &M.z.z);

        constexpr Matrix M1;
        M = M1;
        CHECK(M.m11() == 0);
        CHECK(M.m12() == 0);
        CHECK(M.m13() == 0);
        CHECK(M.m21() == 0);
        CHECK(M.m22() == 0);
        CHECK(M.m23() == 0);
        CHECK(M.m31() == 0);
        CHECK(M.m32() == 0);
        CHECK(M.m33() == 0);

        constexpr Matrix M2{ 1 };
        M = M2;
        CHECK(M.m11() == 1);
        CHECK(M.m12() == 1);
        CHECK(M.m13() == 1);
        CHECK(M.m21() == 1);
        CHECK(M.m22() == 1);
        CHECK(M.m23() == 1);
        CHECK(M.m31() == 1);
        CHECK(M.m32() == 1);
        CHECK(M.m33() == 1);

        constexpr Matrix M3{ 1, 2, 3 };
        M = M3;
        CHECK(M.m11() == 1);
        CHECK(M.m12() == 0);
        CHECK(M.m13() == 0);
        CHECK(M.m21() == 0);
        CHECK(M.m22() == 2);
        CHECK(M.m23() == 0);
        CHECK(M.m31() == 0);
        CHECK(M.m32() == 0);
        CHECK(M.m33() == 3);

        constexpr Matrix M4{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        M = M4;
        CHECK(M.m11() == 1);
        CHECK(M.m12() == 2);
        CHECK(M.m13() == 3);
        CHECK(M.m21() == 4);
        CHECK(M.m22() == 5);
        CHECK(M.m23() == 6);
        CHECK(M.m31() == 7);
        CHECK(M.m32() == 8);
        CHECK(M.m33() == 9);

        constexpr Tensor T{ 1, 2, 3, 4, 5, 6 };
        M = Matrix{ T };
        CHECK(M.x.x == Approx{ T.xx }.epsilon(1e-10));
        CHECK(M.y.y == Approx{ T.yy }.epsilon(1e-10));
        CHECK(M.z.z == Approx{ T.zz }.epsilon(1e-10));
        CHECK(M.x.y == Approx{ T.xy }.epsilon(1e-10));
        CHECK(M.y.z == Approx{ T.yz }.epsilon(1e-10));
        CHECK(M.z.x == Approx{ T.zx }.epsilon(1e-10));

        constexpr auto eye = Matrix::identity();
        CHECK(eye.m11() == 1);
        CHECK(eye.m12() == 0);
        CHECK(eye.m13() == 0);
        CHECK(eye.m21() == 0);
        CHECK(eye.m22() == 1);
        CHECK(eye.m23() == 0);
        CHECK(eye.m31() == 0);
        CHECK(eye.m32() == 0);
        CHECK(eye.m33() == 1);
    }

    {
        constexpr Matrix M1{
            { 0.8058286680786, -0.9554178347078217, 0.37485266605561884 },
            { 0.822599519241102, 0.7627768866338767, -0.9383613170667897 },
            { -0.10307742936493725, -0.17632142314816246, -0.9514092123067659 }
        };
        constexpr auto plus_M1 = +M1;
        CHECK(plus_M1.m11() == Approx{ M1.m11() }.epsilon(1e-10));
        CHECK(plus_M1.m12() == Approx{ M1.m12() }.epsilon(1e-10));
        CHECK(plus_M1.m13() == Approx{ M1.m13() }.epsilon(1e-10));
        CHECK(plus_M1.m21() == Approx{ M1.m21() }.epsilon(1e-10));
        CHECK(plus_M1.m22() == Approx{ M1.m22() }.epsilon(1e-10));
        CHECK(plus_M1.m23() == Approx{ M1.m23() }.epsilon(1e-10));
        CHECK(plus_M1.m31() == Approx{ M1.m31() }.epsilon(1e-10));
        CHECK(plus_M1.m32() == Approx{ M1.m32() }.epsilon(1e-10));
        CHECK(plus_M1.m33() == Approx{ M1.m33() }.epsilon(1e-10));

        constexpr auto minus_M1 = -M1;
        CHECK(minus_M1.m11() == Approx{ -M1.m11() }.epsilon(1e-10));
        CHECK(minus_M1.m12() == Approx{ -M1.m12() }.epsilon(1e-10));
        CHECK(minus_M1.m13() == Approx{ -M1.m13() }.epsilon(1e-10));
        CHECK(minus_M1.m21() == Approx{ -M1.m21() }.epsilon(1e-10));
        CHECK(minus_M1.m22() == Approx{ -M1.m22() }.epsilon(1e-10));
        CHECK(minus_M1.m23() == Approx{ -M1.m23() }.epsilon(1e-10));
        CHECK(minus_M1.m31() == Approx{ -M1.m31() }.epsilon(1e-10));
        CHECK(minus_M1.m32() == Approx{ -M1.m32() }.epsilon(1e-10));
        CHECK(minus_M1.m33() == Approx{ -M1.m33() }.epsilon(1e-10));

        constexpr Matrix M2{
            { 0.041571960178349965, -0.29741806080031985, 0.27928880923747723 },
            { -0.36413955379729623, 0.6318209056262094, -0.9862029820676721 },
            { -0.3888486339500088, 0.6589057836392209, -0.7479051121019191 }
        };
        constexpr Matrix M1_plus_M2_ref{
            { 0.84740062825695, -1.2528358955081416, 0.6541414752930961 },
            { 0.45845996544380574, 1.394597792260086, -1.9245642991344618 },
            { -0.49192606331494604, 0.48258436049105846, -1.699314324408685 }
        };
        constexpr auto M1_plus_M2 = M1 + M2;
        CHECK(M1_plus_M2.m11() == Approx{ M1_plus_M2_ref.m11() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m12() == Approx{ M1_plus_M2_ref.m12() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m13() == Approx{ M1_plus_M2_ref.m13() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m21() == Approx{ M1_plus_M2_ref.m21() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m22() == Approx{ M1_plus_M2_ref.m22() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m23() == Approx{ M1_plus_M2_ref.m23() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m31() == Approx{ M1_plus_M2_ref.m31() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m32() == Approx{ M1_plus_M2_ref.m32() }.epsilon(1e-10));
        CHECK(M1_plus_M2.m33() == Approx{ M1_plus_M2_ref.m33() }.epsilon(1e-10));

        constexpr Matrix M1_minus_M2_ref{
            { 0.76425670790025, -0.6579997739075019, 0.09556385681814161 },
            { 1.1867390730383982, 0.13095598100766725, 0.04784166500088238 },
            { 0.28577120458507155, -0.8352272067873834, -0.20350410020484677 }
        };
        constexpr auto M1_minus_M2 = M1 - M2;
        CHECK(M1_minus_M2.m11() == Approx{ M1_minus_M2_ref.m11() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m12() == Approx{ M1_minus_M2_ref.m12() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m13() == Approx{ M1_minus_M2_ref.m13() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m21() == Approx{ M1_minus_M2_ref.m21() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m22() == Approx{ M1_minus_M2_ref.m22() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m23() == Approx{ M1_minus_M2_ref.m23() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m31() == Approx{ M1_minus_M2_ref.m31() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m32() == Approx{ M1_minus_M2_ref.m32() }.epsilon(1e-10));
        CHECK(M1_minus_M2.m33() == Approx{ M1_minus_M2_ref.m33() }.epsilon(1e-10));

        constexpr Matrix M1_times_M2_ref{
            { 0.03349987729993635, 0.28415851965284084, 0.10469215474216749 },
            { -0.2995410218903253, 0.48193838330375643, 0.9254147291482164 },
            { 0.04008151759963437, -0.11617920549182266, 0.7115638135850904 }
        };
        constexpr auto M1_times_M2 = M1 * M2;
        CHECK(M1_times_M2.m11() == Approx{ M1_times_M2_ref.m11() }.epsilon(1e-10));
        CHECK(M1_times_M2.m12() == Approx{ M1_times_M2_ref.m12() }.epsilon(1e-10));
        CHECK(M1_times_M2.m13() == Approx{ M1_times_M2_ref.m13() }.epsilon(1e-10));
        CHECK(M1_times_M2.m21() == Approx{ M1_times_M2_ref.m21() }.epsilon(1e-10));
        CHECK(M1_times_M2.m22() == Approx{ M1_times_M2_ref.m22() }.epsilon(1e-10));
        CHECK(M1_times_M2.m23() == Approx{ M1_times_M2_ref.m23() }.epsilon(1e-10));
        CHECK(M1_times_M2.m31() == Approx{ M1_times_M2_ref.m31() }.epsilon(1e-10));
        CHECK(M1_times_M2.m32() == Approx{ M1_times_M2_ref.m32() }.epsilon(1e-10));
        CHECK(M1_times_M2.m33() == Approx{ M1_times_M2_ref.m33() }.epsilon(1e-10));

        constexpr Matrix M1_div_M2_ref{
            { 19.383946886831264, 3.2123732907708953, 1.3421685855550494 },
            { -2.2590227034193995, 1.2072675656052791, 0.9514890282519958 },
            { 0.26508368646651614, -0.26759732199392255, 1.2720988223130567 }
        };
        constexpr auto M1_div_M2 = M1 / M2;
        CHECK(M1_div_M2.m11() == Approx{ M1_div_M2_ref.m11() }.epsilon(1e-10));
        CHECK(M1_div_M2.m12() == Approx{ M1_div_M2_ref.m12() }.epsilon(1e-10));
        CHECK(M1_div_M2.m13() == Approx{ M1_div_M2_ref.m13() }.epsilon(1e-10));
        CHECK(M1_div_M2.m21() == Approx{ M1_div_M2_ref.m21() }.epsilon(1e-10));
        CHECK(M1_div_M2.m22() == Approx{ M1_div_M2_ref.m22() }.epsilon(1e-10));
        CHECK(M1_div_M2.m23() == Approx{ M1_div_M2_ref.m23() }.epsilon(1e-10));
        CHECK(M1_div_M2.m31() == Approx{ M1_div_M2_ref.m31() }.epsilon(1e-10));
        CHECK(M1_div_M2.m32() == Approx{ M1_div_M2_ref.m32() }.epsilon(1e-10));
        CHECK(M1_div_M2.m33() == Approx{ M1_div_M2_ref.m33() }.epsilon(1e-10));
    }
    {
        constexpr auto is_equal = [](Matrix A, Matrix B) {
            return A.x.x == B.x.x && A.x.y == B.x.y && A.x.z == B.x.z
                && A.y.x == B.y.x && A.y.y == B.y.y && A.y.z == B.y.z
                && A.z.x == B.z.x && A.z.y == B.z.y && A.z.z == B.z.z;
        };

        constexpr Matrix M1{ { 1, 2, 3 }, { -1, -2, -3 }, { 10, 20, 30 } };
        constexpr double x{ 1.1 };
        CHECK(is_equal(M1 + x, { { 1 + x, 2 + x, 3 + x }, { -1 + x, -2 + x, -3 + x }, { 10 + x, 20 + x, 30 + x } }));
        CHECK(is_equal(M1 - x, { { 1 - x, 2 - x, 3 - x }, { -1 - x, -2 - x, -3 - x }, { 10 - x, 20 - x, 30 - x } }));
        CHECK(is_equal(M1 * x, { { 1 * x, 2 * x, 3 * x }, { -1 * x, -2 * x, -3 * x }, { 10 * x, 20 * x, 30 * x } }));
        CHECK(is_equal(M1 / x, { { 1 / x, 2 / x, 3 / x }, { -1 / x, -2 / x, -3 / x }, { 10 / x, 20 / x, 30 / x } }));
        CHECK(is_equal(x + M1, M1 + x));
        CHECK(is_equal(x - M1, -(M1 - x)));
        CHECK(is_equal(x * M1, M1 * x));
        CHECK(is_equal(x / M1, { { x / 1, x / 2, x / 3 }, { x / -1, x / -2, x / -3 }, { x / 10, x / 20, x / 30 } }));
    }

    {
        constexpr Matrix M1{
            { 0.8058286680786, -0.9554178347078217, 0.37485266605561884 },
            { 0.822599519241102, 0.7627768866338767, -0.9383613170667897 },
            { -0.10307742936493725, -0.17632142314816246, -0.9514092123067659 }
        };
        constexpr Matrix Mtranspose_ref{
            { 0.8058286680786, 0.822599519241102, -0.10307742936493725 },
            { -0.9554178347078217, 0.7627768866338767, -0.17632142314816246 },
            { 0.37485266605561884, -0.9383613170667897, -0.9514092123067659 }
        };
        constexpr auto Mtr = transpose(M1);
        CHECK(Mtr.m11() == Approx{ Mtranspose_ref.m11() }.epsilon(1e-10));
        CHECK(Mtr.m12() == Approx{ Mtranspose_ref.m12() }.epsilon(1e-10));
        CHECK(Mtr.m13() == Approx{ Mtranspose_ref.m13() }.epsilon(1e-10));
        CHECK(Mtr.m21() == Approx{ Mtranspose_ref.m21() }.epsilon(1e-10));
        CHECK(Mtr.m22() == Approx{ Mtranspose_ref.m22() }.epsilon(1e-10));
        CHECK(Mtr.m23() == Approx{ Mtranspose_ref.m23() }.epsilon(1e-10));
        CHECK(Mtr.m31() == Approx{ Mtranspose_ref.m31() }.epsilon(1e-10));
        CHECK(Mtr.m32() == Approx{ Mtranspose_ref.m32() }.epsilon(1e-10));
        CHECK(Mtr.m33() == Approx{ Mtranspose_ref.m33() }.epsilon(1e-10));
        CHECK(trace(M1) == Approx{ 0.6171963424057108 }.epsilon(1e-10));
        CHECK(det(M1) == Approx{ -1.5831729566386807 }.epsilon(1e-10));

        constexpr Matrix Minv_ref{
            { 0.562898801418783, 0.6159073656508607, -0.3856800266861635 },
            { -0.5554368708336475, 0.4598575070970089, -0.6723915215505777 },
            { 0.04195172547754246, -0.15195236308928883, -0.8846751254998837 }
        };
        constexpr auto Minv = inv(M1);
        CHECK(Minv.m11() == Approx{ Minv_ref.m11() }.epsilon(1e-10));
        CHECK(Minv.m12() == Approx{ Minv_ref.m12() }.epsilon(1e-10));
        CHECK(Minv.m13() == Approx{ Minv_ref.m13() }.epsilon(1e-10));
        CHECK(Minv.m21() == Approx{ Minv_ref.m21() }.epsilon(1e-10));
        CHECK(Minv.m22() == Approx{ Minv_ref.m22() }.epsilon(1e-10));
        CHECK(Minv.m23() == Approx{ Minv_ref.m23() }.epsilon(1e-10));
        CHECK(Minv.m31() == Approx{ Minv_ref.m31() }.epsilon(1e-10));
        CHECK(Minv.m32() == Approx{ Minv_ref.m32() }.epsilon(1e-10));
        CHECK(Minv.m33() == Approx{ Minv_ref.m33() }.epsilon(1e-10));

        constexpr Matrix M2{
            { 0.041571960178349965, -0.29741806080031985, 0.27928880923747723 },
            { -0.36413955379729623, 0.6318209056262094, -0.9862029820676721 },
            { -0.3888486339500088, 0.6589057836392209, -0.7479051121019191 }
        };
        constexpr Matrix M1_dot_M2_ref{
            { 0.23564435419217528, -0.596328371697141, 0.8869406216385556 },
            { 0.12132035560382248, -0.3810092694828321, 0.17929522599048664 },
            { 0.4298746460941701, -0.7076355046973359, 0.8566641543896305 }
        };
        constexpr auto M1_dot_M2 = dot(M1, M2);
        CHECK(M1_dot_M2.m11() == Approx{ M1_dot_M2_ref.m11() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m12() == Approx{ M1_dot_M2_ref.m12() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m13() == Approx{ M1_dot_M2_ref.m13() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m21() == Approx{ M1_dot_M2_ref.m21() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m22() == Approx{ M1_dot_M2_ref.m22() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m23() == Approx{ M1_dot_M2_ref.m23() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m31() == Approx{ M1_dot_M2_ref.m31() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m32() == Approx{ M1_dot_M2_ref.m32() }.epsilon(1e-10));
        CHECK(M1_dot_M2.m33() == Approx{ M1_dot_M2_ref.m33() }.epsilon(1e-10));

        constexpr Tensor T{
            -0.798004311337166, 0.640423661686853, 0.43654971327364933,
            0.19639357333284924, -0.8383142711428451, 0.37525093367772033
        };
        constexpr Matrix M1_dot_T_ref{
            { -0.6900288609810579, -0.7678569560568693, 1.2669701897071748 },
            { -0.8587544447929149, 1.4366953093792947, -0.7404068760376106 },
            { -0.3093893565401272, 0.6644157642442071, -0.30620455511539824 }
        };
        constexpr auto M1_dot_T = dot(M1, T);
        CHECK(M1_dot_T.m11() == Approx{ M1_dot_T_ref.m11() }.epsilon(1e-10));
        CHECK(M1_dot_T.m12() == Approx{ M1_dot_T_ref.m12() }.epsilon(1e-10));
        CHECK(M1_dot_T.m13() == Approx{ M1_dot_T_ref.m13() }.epsilon(1e-10));
        CHECK(M1_dot_T.m21() == Approx{ M1_dot_T_ref.m21() }.epsilon(1e-10));
        CHECK(M1_dot_T.m22() == Approx{ M1_dot_T_ref.m22() }.epsilon(1e-10));
        CHECK(M1_dot_T.m23() == Approx{ M1_dot_T_ref.m23() }.epsilon(1e-10));
        CHECK(M1_dot_T.m31() == Approx{ M1_dot_T_ref.m31() }.epsilon(1e-10));
        CHECK(M1_dot_T.m32() == Approx{ M1_dot_T_ref.m32() }.epsilon(1e-10));
        CHECK(M1_dot_T.m33() == Approx{ M1_dot_T_ref.m33() }.epsilon(1e-10));

        constexpr Matrix T_dot_M2_ref{
            { -0.25060508456418745, 0.6086814706667905, -0.6972096931780973 },
            { 0.10093823855668835, -0.20614805958617943, 0.050242331231480364 },
            { 0.1511115418498735, -0.3536257560787339, 0.6050536582990735 }
        };
        constexpr auto T_dot_M2 = dot(T, M2);
        CHECK(T_dot_M2.m11() == Approx{ T_dot_M2_ref.m11() }.epsilon(1e-10));
        CHECK(T_dot_M2.m12() == Approx{ T_dot_M2_ref.m12() }.epsilon(1e-10));
        CHECK(T_dot_M2.m13() == Approx{ T_dot_M2_ref.m13() }.epsilon(1e-10));
        CHECK(T_dot_M2.m21() == Approx{ T_dot_M2_ref.m21() }.epsilon(1e-10));
        CHECK(T_dot_M2.m22() == Approx{ T_dot_M2_ref.m22() }.epsilon(1e-10));
        CHECK(T_dot_M2.m23() == Approx{ T_dot_M2_ref.m23() }.epsilon(1e-10));
        CHECK(T_dot_M2.m31() == Approx{ T_dot_M2_ref.m31() }.epsilon(1e-10));
        CHECK(T_dot_M2.m32() == Approx{ T_dot_M2_ref.m32() }.epsilon(1e-10));
        CHECK(T_dot_M2.m33() == Approx{ T_dot_M2_ref.m33() }.epsilon(1e-10));

        constexpr Vector V{ 0.6811089201530276, -0.9218070818531872, 0.33572786357923334 };
        constexpr Vector M1_dot_V_ref{ 1.5554165048377084, -0.4578873059485554, -0.2270872023038455 };
        constexpr auto   M1_dot_V = dot(M1, V);
        CHECK(M1_dot_V.x == Approx{ M1_dot_V_ref.x }.epsilon(1e-10));
        CHECK(M1_dot_V.y == Approx{ M1_dot_V_ref.y }.epsilon(1e-10));
        CHECK(M1_dot_V.z == Approx{ M1_dot_V_ref.z }.epsilon(1e-10));

        constexpr Vector V_dot_M2_ref{ 0.23343413124718812, -0.5637780484536513, 0.8482224068393107 };
        constexpr Vector V_dot_M2 = dot(V, M2);
        CHECK(V_dot_M2.x == Approx{ V_dot_M2_ref.x }.epsilon(1e-10));
        CHECK(V_dot_M2.y == Approx{ V_dot_M2_ref.y }.epsilon(1e-10));
        CHECK(V_dot_M2.z == Approx{ V_dot_M2_ref.z }.epsilon(1e-10));
    }
}
