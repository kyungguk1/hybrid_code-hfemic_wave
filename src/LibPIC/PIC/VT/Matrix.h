/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/VT/Tensor.h>
#include <PIC/VT/Vector.h>

#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
/// 3x3 matrix
///
struct alignas(Vector) Matrix {
    using Real = double;

    // matrix elements
    //
    Vector x; // 1st row
    Vector y; // 2nd row
    Vector z; // 3rd row

    [[nodiscard]] constexpr Real &m11() noexcept { return x.x; }
    [[nodiscard]] constexpr Real &m12() noexcept { return x.y; }
    [[nodiscard]] constexpr Real &m13() noexcept { return x.z; }
    [[nodiscard]] constexpr Real &m21() noexcept { return y.x; }
    [[nodiscard]] constexpr Real &m22() noexcept { return y.y; }
    [[nodiscard]] constexpr Real &m23() noexcept { return y.z; }
    [[nodiscard]] constexpr Real &m31() noexcept { return z.x; }
    [[nodiscard]] constexpr Real &m32() noexcept { return z.y; }
    [[nodiscard]] constexpr Real &m33() noexcept { return z.z; }

    [[nodiscard]] constexpr Real const &m11() const noexcept { return x.x; }
    [[nodiscard]] constexpr Real const &m12() const noexcept { return x.y; }
    [[nodiscard]] constexpr Real const &m13() const noexcept { return x.z; }
    [[nodiscard]] constexpr Real const &m21() const noexcept { return y.x; }
    [[nodiscard]] constexpr Real const &m22() const noexcept { return y.y; }
    [[nodiscard]] constexpr Real const &m23() const noexcept { return y.z; }
    [[nodiscard]] constexpr Real const &m31() const noexcept { return z.x; }
    [[nodiscard]] constexpr Real const &m32() const noexcept { return z.y; }
    [[nodiscard]] constexpr Real const &m33() const noexcept { return z.z; }

    // constructors
    //
    constexpr Matrix() noexcept = default;
    constexpr explicit Matrix(Real fill) noexcept
    : x{ fill }, y{ fill }, z{ fill } {}
    constexpr explicit Matrix(Real xx, Real yy, Real zz) noexcept
    : x{ xx, 0, 0 }, y{ 0, yy, 0 }, z{ 0, 0, zz } {}
    constexpr Matrix(Vector row1, Vector row2, Vector row3) noexcept
    : x{ row1 }, y{ row2 }, z{ row3 } {}
    constexpr explicit Matrix(Tensor t) noexcept
    : x{ t.xx, t.xy, t.zx }, y{ t.xy, t.yy, t.yz }, z{ t.zx, t.yz, t.zz } {}

    [[nodiscard]] static constexpr auto identity() noexcept { return Matrix{ 1, 1, 1 }; }

    // matrix calculus
    //
    [[nodiscard]] friend constexpr Matrix transpose(Matrix const &A) noexcept
    {
        return {
            { A.x.x, A.y.x, A.z.x },
            { A.x.y, A.y.y, A.z.y },
            { A.x.z, A.y.z, A.z.z },
        };
    }
    [[nodiscard]] friend constexpr Real trace(Matrix const &A) noexcept
    {
        return A.x.x + A.y.y + A.z.z;
    }
    [[nodiscard]] friend constexpr Real det(Matrix const &A) noexcept
    {
        return (A.x.x * A.y.y * A.z.z + A.x.y * A.y.z * A.z.x + A.x.z * A.y.x * A.z.y)
             - (A.x.z * A.y.y * A.z.x + A.x.x * A.y.z * A.z.y + A.x.y * A.y.x * A.z.z);
    }
    [[nodiscard]] friend constexpr Matrix inv(Matrix const &A) noexcept
    {
        Matrix inv{
            { A.y.y * A.z.z - A.y.z * A.z.y, A.x.z * A.z.y - A.x.y * A.z.z, A.x.y * A.y.z - A.x.z * A.y.y },
            { A.y.z * A.z.x - A.y.x * A.z.z, A.x.x * A.z.z - A.x.z * A.z.x, A.x.z * A.y.x - A.x.x * A.y.z },
            { A.y.x * A.z.y - A.y.y * A.z.x, A.x.y * A.z.x - A.x.x * A.z.y, A.x.x * A.y.y - A.x.y * A.y.x },
        };
        inv /= det(A);
        return inv;
    }
    [[nodiscard]] friend constexpr Matrix dot(Matrix const &A, Matrix const &B) noexcept
    {
        return {
            { A.x.x * B.x.x + A.x.y * B.y.x + A.x.z * B.z.x, A.x.x * B.x.y + A.x.y * B.y.y + A.x.z * B.z.y, A.x.x * B.x.z + A.x.y * B.y.z + A.x.z * B.z.z },
            { A.y.x * B.x.x + A.y.y * B.y.x + A.y.z * B.z.x, A.y.x * B.x.y + A.y.y * B.y.y + A.y.z * B.z.y, A.y.x * B.x.z + A.y.y * B.y.z + A.y.z * B.z.z },
            { A.z.x * B.x.x + A.z.y * B.y.x + A.z.z * B.z.x, A.z.x * B.x.y + A.z.y * B.y.y + A.z.z * B.z.y, A.z.x * B.x.z + A.z.y * B.y.z + A.z.z * B.z.z },
        };
    }
    [[nodiscard]] friend constexpr Matrix dot(Matrix const &A, Tensor const &B) noexcept
    {
        return {
            { A.x.x * B.xx + A.x.y * B.xy + A.x.z * B.zx, A.x.x * B.xy + A.x.y * B.yy + A.x.z * B.yz, A.x.x * B.zx + A.x.y * B.yz + A.x.z * B.zz },
            { A.y.x * B.xx + A.y.y * B.xy + A.y.z * B.zx, A.y.x * B.xy + A.y.y * B.yy + A.y.z * B.yz, A.y.x * B.zx + A.y.y * B.yz + A.y.z * B.zz },
            { A.z.x * B.xx + A.z.y * B.xy + A.z.z * B.zx, A.z.x * B.xy + A.z.y * B.yy + A.z.z * B.yz, A.z.x * B.zx + A.z.y * B.yz + A.z.z * B.zz },
        };
    }
    [[nodiscard]] friend constexpr Matrix dot(Tensor const &A, Matrix const &B) noexcept
    {
        return {
            { A.xx * B.x.x + A.xy * B.y.x + A.zx * B.z.x, A.xx * B.x.y + A.xy * B.y.y + A.zx * B.z.y, A.xx * B.x.z + A.xy * B.y.z + A.zx * B.z.z },
            { A.xy * B.x.x + A.yy * B.y.x + A.yz * B.z.x, A.xy * B.x.y + A.yy * B.y.y + A.yz * B.z.y, A.xy * B.x.z + A.yy * B.y.z + A.yz * B.z.z },
            { A.zx * B.x.x + A.yz * B.y.x + A.zz * B.z.x, A.zx * B.x.y + A.yz * B.y.y + A.zz * B.z.y, A.zx * B.x.z + A.yz * B.y.z + A.zz * B.z.z },
        };
    }
    [[nodiscard]] friend constexpr Vector dot(Matrix const &A, Vector const &b) noexcept
    {
        return { dot(A.x, b), dot(A.y, b), dot(A.z, b) };
    }
    [[nodiscard]] friend constexpr Vector dot(Vector const &a, Matrix const &B) noexcept
    {
        return { a.x * B.x.x + a.y * B.y.x + a.z * B.z.x, a.x * B.x.y + a.y * B.y.y + a.z * B.z.y, a.x * B.x.z + a.y * B.y.z + a.z * B.z.z };
    }

    // compound operations: matrix @= matrix, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    constexpr Matrix &operator+=(Matrix const &o) noexcept
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
    constexpr Matrix &operator-=(Matrix const &o) noexcept
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }
    constexpr Matrix &operator*=(Matrix const &o) noexcept
    {
        x *= o.x;
        y *= o.y;
        z *= o.z;
        return *this;
    }
    constexpr Matrix &operator/=(Matrix const &o) noexcept
    {
        x /= o.x;
        y /= o.y;
        z /= o.z;
        return *this;
    }

    // scalar-matrix compound operations: matrix @= real, where @ is one of +, -, *, and /
    // operation with scalar is distributed to all elements
    //
    constexpr Matrix &operator+=(Real const &s) noexcept
    {
        x += s;
        y += s;
        z += s;
        return *this;
    }
    constexpr Matrix &operator-=(Real const &s) noexcept
    {
        x -= s;
        y -= s;
        z -= s;
        return *this;
    }
    constexpr Matrix &operator*=(Real const &s) noexcept
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    constexpr Matrix &operator/=(Real const &s) noexcept
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr Matrix const &operator+(Matrix const &v) noexcept { return v; }
    [[nodiscard]] friend constexpr Matrix        operator-(Matrix v) noexcept
    {
        v *= Real{ -1 };
        return v;
    }

    // binary operations: matrix @ {matrix|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr Matrix operator+(Matrix a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr Matrix operator-(Matrix a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr Matrix operator*(Matrix a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr Matrix operator/(Matrix a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ matrix, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr Matrix operator+(Real const &b, Matrix const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr Matrix operator-(Real const &a, Matrix const &b) noexcept
    {
        Matrix A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr Matrix operator*(Real const &b, Matrix const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr Matrix operator/(Real const &a, Matrix const &b) noexcept
    {
        Matrix A{ a };
        A /= b;
        return A;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Matrix const &A)
    {
        os << '{';
        os << '{' << A.x.x << ", " << A.x.y << ", " << A.x.z << "}, ";
        os << '{' << A.y.x << ", " << A.y.y << ", " << A.y.z << "}, ";
        os << '{' << A.z.x << ", " << A.z.y << ", " << A.z.z << '}';
        os << '}';
        return os;
    }
};

static_assert(std::is_standard_layout_v<Matrix>);
LIBPIC_NAMESPACE_END(1)
