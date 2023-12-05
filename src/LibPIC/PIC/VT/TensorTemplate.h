/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

#include <functional>
#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
/// Compact symmetric rank-2 tensor template
///
template <class ConcreteTensor, class Vector>
struct alignas(Vector) TensorTemplate {
    using ElementType = typename Vector::ElementType;

    // tensor elements
    //
    ElementType xx{}, yy{}, zz{}; // diagonal
    ElementType xy{}, yz{}, zx{}; // off-diag

    // constructors
    //
    constexpr TensorTemplate() noexcept = default;
    constexpr explicit TensorTemplate(ElementType const &v) noexcept
    : TensorTemplate{ v, v, v, v, v, v }
    {
    }
    constexpr TensorTemplate(
        ElementType const &xx, ElementType const &yy, ElementType const &zz,
        ElementType const &xy, ElementType const &yz, ElementType const &zx) noexcept
    : xx{ xx }, yy{ yy }, zz{ zz }, xy{ xy }, yz{ yz }, zx{ zx }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Row index.
    /// \tparam J Column index.
    /// \param vv Tensor.
    /// \return Indexed value.
    template <long I, long J>
    [[nodiscard]] constexpr friend auto &get(ConcreteTensor const &vv) noexcept
    {
        static_assert(I >= 0 && I < 3, "row index out of range");
        static_assert(J >= 0 && J < 3, "column index out of range");
        return impl_get<I, J>(vv);
    }
    template <long I, long J>
    [[nodiscard]] constexpr friend auto &get(ConcreteTensor &vv) noexcept
    {
        static_assert(I >= 0 && I < 3, "row index out of range");
        static_assert(J >= 0 && J < 3, "column index out of range");
        return impl_get<I, J>(vv);
    }

    // access to lower and upper halves as a vector
    //
    [[nodiscard]] Vector       &lo() noexcept { return *reinterpret_cast<Vector *>(&xx); }
    [[nodiscard]] Vector const &lo() const noexcept { return *reinterpret_cast<Vector const *>(&xx); }

    [[nodiscard]] Vector       &hi() noexcept { return *reinterpret_cast<Vector *>(&xy); }
    [[nodiscard]] Vector const &hi() const noexcept { return *reinterpret_cast<Vector const *>(&xy); }

    // linear algebra
    //
    [[nodiscard]] friend constexpr auto trace(ConcreteTensor const &A) noexcept -> ElementType
    {
        return A.xx + A.yy + A.zz;
    }
    [[nodiscard]] friend constexpr auto transpose(ConcreteTensor const &A) noexcept -> decltype(auto)
    {
        return A;
    }
    [[nodiscard]] friend constexpr auto det(ConcreteTensor const &A) noexcept -> ElementType
    {
        return (A.xx * A.yy * A.zz + 2 * A.xy * A.yz * A.zx)
             - (A.xx * A.yz * A.yz + A.yy * A.zx * A.zx + A.xy * A.xy * A.zz);
    }
    [[nodiscard]] friend constexpr auto inv(ConcreteTensor const &A) noexcept -> ConcreteTensor
    {
        ConcreteTensor inv{
            A.yy * A.zz - A.yz * A.yz, A.xx * A.zz - A.zx * A.zx, A.xx * A.yy - A.xy * A.xy,
            A.yz * A.zx - A.xy * A.zz, A.xy * A.zx - A.xx * A.yz, A.xy * A.yz - A.yy * A.zx
        };
        inv /= det(A);
        return inv;
    }

    // left-fold: applies to all elements
    // the signature of BinaryOp is "Init(Init, Type)"
    //
    template <class Init, class BinaryOp,
              std::enable_if_t<std::is_invocable_r_v<Init, BinaryOp, Init, ElementType>, int> = 0>
    [[nodiscard]] constexpr auto fold(Init init, BinaryOp &&f) const
        noexcept(std::is_nothrow_invocable_r_v<Init, BinaryOp, Init, ElementType>)
    {
        return f(f(f(f(f(f(init, xx), yy), zz), xy), yz), zx);
    }

    // compound operations: tensor @= tensor, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(ConcreteTensor &lhs, ConcreteTensor const &rhs) noexcept
    {
        lhs.xx += rhs.xx;
        lhs.yy += rhs.yy;
        lhs.zz += rhs.zz;
        lhs.xy += rhs.xy;
        lhs.yz += rhs.yz;
        lhs.zx += rhs.zx;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteTensor &lhs, ConcreteTensor const &rhs) noexcept
    {
        lhs.xx -= rhs.xx;
        lhs.yy -= rhs.yy;
        lhs.zz -= rhs.zz;
        lhs.xy -= rhs.xy;
        lhs.yz -= rhs.yz;
        lhs.zx -= rhs.zx;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteTensor &lhs, ConcreteTensor const &rhs) noexcept
    {
        lhs.xx *= rhs.xx;
        lhs.yy *= rhs.yy;
        lhs.zz *= rhs.zz;
        lhs.xy *= rhs.xy;
        lhs.yz *= rhs.yz;
        lhs.zx *= rhs.zx;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteTensor &lhs, ConcreteTensor const &rhs) noexcept
    {
        lhs.xx /= rhs.xx;
        lhs.yy /= rhs.yy;
        lhs.zz /= rhs.zz;
        lhs.xy /= rhs.xy;
        lhs.yz /= rhs.yz;
        lhs.zx /= rhs.zx;
        return lhs;
    }

    // scalar-tensor compound operations: tensor @= real, where @ is one of +, -, *, and /
    // operation with scalar is distributed to all elements
    //
    friend constexpr decltype(auto) operator+=(ConcreteTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.xx += rhs;
        lhs.yy += rhs;
        lhs.zz += rhs;
        lhs.xy += rhs;
        lhs.yz += rhs;
        lhs.zx += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.xx -= rhs;
        lhs.yy -= rhs;
        lhs.zz -= rhs;
        lhs.xy -= rhs;
        lhs.yz -= rhs;
        lhs.zx -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.xx *= rhs;
        lhs.yy *= rhs;
        lhs.zz *= rhs;
        lhs.xy *= rhs;
        lhs.yz *= rhs;
        lhs.zx *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.xx /= rhs;
        lhs.yy /= rhs;
        lhs.zz /= rhs;
        lhs.xy /= rhs;
        lhs.yz /= rhs;
        lhs.zx /= rhs;
        return lhs;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(ConcreteTensor const &v) noexcept { return v; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(ConcreteTensor const &v) noexcept { return ConcreteTensor{} - v; }

    // binary operations: tensor @ {tensor|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr auto operator+(ConcreteTensor a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator-(ConcreteTensor a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator*(ConcreteTensor a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator/(ConcreteTensor a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ tensor, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr auto operator+(ElementType const &b, ConcreteTensor const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr auto operator-(ElementType const &a, ConcreteTensor const &b) noexcept
    {
        ConcreteTensor A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr auto operator*(ElementType const &b, ConcreteTensor const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr auto operator/(ElementType const &a, ConcreteTensor const &b) noexcept
    {
        ConcreteTensor A{ a };
        A /= b;
        return A;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, ConcreteTensor const &v)
    {
        return os << '{' << v.xx << ", " << v.yy << ", " << v.zz
                  << ", " << v.xy << ", " << v.yz << ", " << v.zx << '}';
    }

private:
    template <long I, long J, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &vv) noexcept
    {
        if constexpr (I == 0) {
            if constexpr (J == 0)
                return vv.xx;
            else if constexpr (J == 1)
                return vv.xy;
            else if constexpr (J == 2)
                return vv.zx;
        } else if constexpr (I == 1) {
            if constexpr (J == 0)
                return vv.xy;
            else if constexpr (J == 1)
                return vv.yy;
            else if constexpr (J == 2)
                return vv.yz;
        } else if constexpr (I == 2) {
            if constexpr (J == 0)
                return vv.zx;
            else if constexpr (J == 1)
                return vv.yz;
            else if constexpr (J == 2)
                return vv.zz;
        }
    }
};

/// Tensor calculus interface
///
template <class ConcreteTensor, class Vector>
struct alignas(Vector) TensorCalculus {
    [[nodiscard]] static constexpr auto identity() noexcept { return ConcreteTensor{ 1, 1, 1, 0, 0, 0 }; }

    [[nodiscard]] friend constexpr auto dot(ConcreteTensor const &A, Vector const &b) noexcept -> Vector
    {
        return {
            A.xx * b.x + A.xy * b.y + A.zx * b.z,
            A.xy * b.x + A.yy * b.y + A.yz * b.z,
            A.zx * b.x + A.yz * b.y + A.zz * b.z,
        };
    }
    [[nodiscard]] friend constexpr auto dot(Vector const &b, ConcreteTensor const &A) noexcept -> Vector
    {
        return dot(A, b); // because A^T == A
    }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
