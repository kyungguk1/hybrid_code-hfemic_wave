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
/// Concrete vector template
///
template <class ConcreteVector, class ElementType>
struct VectorTemplate {
    // vector elements
    //
    ElementType x{};
    ElementType y{};
    ElementType z{};

    // constructors
    //
    constexpr VectorTemplate() noexcept = default;
    constexpr explicit VectorTemplate(ElementType const &v) noexcept
    : VectorTemplate{ v, v, v }
    {
    }
    constexpr VectorTemplate(ElementType const &x, ElementType const &y, ElementType const &z) noexcept
    : x{ x }, y{ y }, z{ z }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Index.
    /// \param v Vector.
    /// \return Indexed value.
    template <long I>
    [[nodiscard]] constexpr friend auto &get(ConcreteVector const &v) noexcept
    {
        static_assert(I >= 0 && I < 3, "index out of range");
        return impl_get<I>(v);
    }
    template <long I>
    [[nodiscard]] constexpr friend auto &get(ConcreteVector &v) noexcept
    {
        static_assert(I >= 0 && I < 3, "index out of range");
        return impl_get<I>(v);
    }

    // left-fold: applies to all elements
    // the signature of BinaryOp is "Init(Init, Type)"
    //
    template <class Init, class BinaryOp,
              std::enable_if_t<std::is_invocable_r_v<Init, BinaryOp, Init, ElementType>, int> = 0>
    [[nodiscard]] constexpr auto fold(Init init, BinaryOp &&f) const
        noexcept(std::is_nothrow_invocable_r_v<Init, BinaryOp, Init, ElementType>)
    {
        return f(f(f(init, x), y), z);
    }

    // compound operations: vector @= vector, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(ConcreteVector &lhs, ConcreteVector const &rhs) noexcept
    {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteVector &lhs, ConcreteVector const &rhs) noexcept
    {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteVector &lhs, ConcreteVector const &rhs) noexcept
    {
        lhs.x *= rhs.x;
        lhs.y *= rhs.y;
        lhs.z *= rhs.z;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteVector &lhs, ConcreteVector const &rhs) noexcept
    {
        lhs.x /= rhs.x;
        lhs.y /= rhs.y;
        lhs.z /= rhs.z;
        return lhs;
    }

    // scalar-vector compound operations: vector @= type, where @ is one of +, -, *, and /
    // operation with scalar is distributed to all elements
    //
    friend constexpr decltype(auto) operator+=(ConcreteVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.x += rhs;
        lhs.y += rhs;
        lhs.z += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.x -= rhs;
        lhs.y -= rhs;
        lhs.z -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.x /= rhs;
        lhs.y /= rhs;
        lhs.z /= rhs;
        return lhs;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(ConcreteVector const &v) noexcept { return v; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(ConcreteVector const &v) noexcept { return ConcreteVector{} - v; }

    // binary operations: vector @ {vector|type}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr auto operator+(ConcreteVector a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator-(ConcreteVector a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator*(ConcreteVector a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator/(ConcreteVector a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: type @ vector, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr auto operator+(ElementType const &b, ConcreteVector const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr auto operator-(ElementType const &a, ConcreteVector const &b) noexcept
    {
        ConcreteVector A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr auto operator*(ElementType const &b, ConcreteVector const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr auto operator/(ElementType const &a, ConcreteVector const &b) noexcept
    {
        ConcreteVector A{ a };
        A /= b;
        return A;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, ConcreteVector const &v)
    {
        return os << '{' << v.x << ", " << v.y << ", " << v.z << '}';
    }

private:
    template <long I, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &v) noexcept
    {
        if constexpr (I == 0)
            return v.x;
        else if constexpr (I == 1)
            return v.y;
        else if constexpr (I == 2)
            return v.z;
    }
};

/// Vector calculus interface
///
template <class ConcreteVector, class ElementType>
struct VectorCalculus {
    [[nodiscard]] friend constexpr auto dot(ConcreteVector const &A, ConcreteVector const &B) noexcept -> ElementType
    {
        return A.x * B.x + A.y * B.y + A.z * B.z;
    }
    [[nodiscard]] friend constexpr auto cross(ConcreteVector const &A, ConcreteVector const &B) noexcept -> ConcreteVector
    {
        return { A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x };
    }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
