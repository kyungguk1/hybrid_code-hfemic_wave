/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/VT/Scalar.h>

#include <cmath>
#include <ostream>
#include <stdexcept>
#include <tuple>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
/// Four-vector template
///
template <class ConcreteFourVector, class Vector>
struct FourVectorTemplate {
    using ElementType = typename Vector::ElementType;

    // vector elements
    //
    Scalar t{}; //!< time component
    Vector s{}; //!< space components

    // constructors
    //
    constexpr FourVectorTemplate() noexcept = default;
    constexpr explicit FourVectorTemplate(ElementType const &v) noexcept
    : t{ v }, s{ v }
    {
    }
    constexpr FourVectorTemplate(Scalar const &t, Vector const &s) noexcept
    : t{ t }, s{ s }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Index.
    /// \param fv FourVector.
    /// \return Indexed value.
    template <long I>
    [[nodiscard]] constexpr friend auto &get(ConcreteFourVector const &fv) noexcept
    {
        static_assert(I >= 0 && I < 4, "index out of range");
        return impl_get<I>(fv);
    }
    template <long I>
    [[nodiscard]] constexpr friend auto &get(ConcreteFourVector &fv) noexcept
    {
        static_assert(I >= 0 && I < 4, "index out of range");
        return impl_get<I>(fv);
    }

    // compound operations: vector @= vector, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(ConcreteFourVector &lhs, ConcreteFourVector const &rhs) noexcept
    {
        lhs.t += rhs.t;
        lhs.s += rhs.s;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteFourVector &lhs, ConcreteFourVector const &rhs) noexcept
    {
        lhs.t -= rhs.t;
        lhs.s -= rhs.s;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteFourVector &lhs, ConcreteFourVector const &rhs) noexcept
    {
        lhs.t *= rhs.t;
        lhs.s *= rhs.s;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteFourVector &lhs, ConcreteFourVector const &rhs) noexcept
    {
        lhs.t /= rhs.t;
        lhs.s /= rhs.s;
        return lhs;
    }

    // compound operations: vector @= real, where @ is one of +, -, *, and /
    // operation with scalar is distributed to all elements
    //
    friend constexpr decltype(auto) operator+=(ConcreteFourVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.t += rhs;
        lhs.s += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteFourVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.t -= rhs;
        lhs.s -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteFourVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.t *= rhs;
        lhs.s *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteFourVector &lhs, ElementType const &rhs) noexcept
    {
        lhs.t /= rhs;
        lhs.s /= rhs;
        return lhs;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(ConcreteFourVector const &F) noexcept { return F; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(ConcreteFourVector const &F) noexcept { return ConcreteFourVector{} - F; }

    // binary operations: vector @ {vector|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr auto operator+(ConcreteFourVector a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator-(ConcreteFourVector a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator*(ConcreteFourVector a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr auto operator/(ConcreteFourVector a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ vector, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr ConcreteFourVector operator+(ElementType const &b, ConcreteFourVector const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr ConcreteFourVector operator-(ElementType const &a, ConcreteFourVector const &b) noexcept
    {
        ConcreteFourVector A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr ConcreteFourVector operator*(ElementType const &b, ConcreteFourVector const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr ConcreteFourVector operator/(ElementType const &a, ConcreteFourVector const &b) noexcept
    {
        ConcreteFourVector A{ a };
        A /= b;
        return A;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, ConcreteFourVector const &v)
    {
        return os << '{' << v.t << ", " << v.s << '}';
    }

private:
    template <long I, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &fv) noexcept
    {
        using std::get;
        if constexpr (I == 0)
            return fv.t;
        else if constexpr (I > 0)
            return get<I - 1>(fv.s);
    }
};

/// Four-vector calculus interface
///
template <class ConcreteFourVector, class Vector>
struct FourVectorBoost {
    using ElementType = typename Vector::ElementType;

    /// Lorentz boost
    /// \details Consider the primed frame of reference is moving relative to the lab reference frame
    ///          at a velocity normalized to the light speed, β = v/c.
    ///          The transformation of a four-vector, (x0, x1, x2, x3), in the lab frame
    ///          to a four-vector, (x0', x1', x2', x3'), in the moving frame is
    ///                x0' = γx0 - γβ·xx;
    ///                xx' = xx + ((γ - 1)/β^2(β·xx) - γx0)β;
    ///          where xx and xx' are the spatial component of the four-vector.
    ///
    /// \tparam dir If +1, forward boost, else if -1, inverse boost (same as if β is replaced with -β).
    /// \param F Contravariant four-vector components.
    /// \param beta β = v/c.
    /// \param gamma Relativistic factor, 1/√(1 - β^2).
    /// \return Contravariant four-vector components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourVector lorentz_boost(ConcreteFourVector const &F, Vector beta, ElementType const &gamma)
    {
        static_assert(dir * dir == 1, "abs(dir) must be 1");
        if constexpr (dir > 0)
            beta *= -1;
        return F.impl_lorentz_boost(beta, gamma);
    }
    /// Lorentz boost
    ///
    /// \tparam dir If +1, forward boost, else if -1, inverse boost (same as if β is replaced with -β).
    /// \param F Contravariant four-vector components.
    /// \param beta_x_gamma γ times β.
    /// \return Contravariant four-vector components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourVector lorentz_boost(ConcreteFourVector const &F, Vector beta_x_gamma)
    {
        static_assert(dir * dir == 1, "abs(dir) must be 1");
        if constexpr (dir > 0)
            beta_x_gamma *= -1;
        auto const gamma = std::sqrt(1 + dot(beta_x_gamma, beta_x_gamma));
        return F.impl_lorentz_boost(beta_x_gamma / gamma, gamma);
    }

    /// Lorentz boost
    /// \details In this case, the primed frame of reference is moving in the x direction relative to the lab reference frame
    ///          at a velocity normalized to the light speed, β = v/c.
    ///          The transformation of a four-vector, (x0, x1, x2, x3), in the lab frame
    ///          to a four-vector, (x0', x1', x2', x3'), in the moving frame is
    ///                x0' = γ(x0 - βx1);
    ///                x1' = γ(x1 - βx0);
    ///                x2' = x2;
    ///                x3' = x3.
    ///
    /// \tparam dir The direction of boost - 1: x-dir, 2: y-dir, 3: z-dir.
    ///             The negative sign indicates inverse boost.
    /// \param F Contravariant four-vector components.
    /// \param beta β = v/c.
    /// \param gamma Relativistic factor, 1/√(1 - β^2).
    /// \return Contravariant four-vector components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourVector lorentz_boost(ConcreteFourVector const &F, ElementType beta, ElementType const &gamma)
    {
        static_assert(dir * dir >= 1 && dir * dir <= 9, "abs(dir) must be between 1 and 3");
        if constexpr (dir > 0)
            beta *= -1;
        return F.template impl_lorentz_boost<(dir < 0 ? -dir : dir)>(beta, gamma);
    }
    /// Lorentz boost
    ///
    /// \tparam dir The direction of boost - 1: x-dir, 2: y-dir, 3: z-dir.
    ///             The negative sign indicates inverse boost.
    /// \param F Contravariant four-vector components.
    /// \param beta_x_gamma γ times β.
    /// \return Contravariant four-vector components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourVector lorentz_boost(ConcreteFourVector const &F, ElementType beta_x_gamma)
    {
        static_assert(dir * dir >= 1 && dir * dir <= 9, "abs(dir) must be between 1 and 3");
        if constexpr (dir > 0)
            beta_x_gamma *= -1;
        auto const gamma = std::sqrt(1 + beta_x_gamma * beta_x_gamma);
        return F.template impl_lorentz_boost<(dir < 0 ? -dir : dir)>(beta_x_gamma / gamma, gamma);
    }

private:
    [[nodiscard]] auto self() const &noexcept { return static_cast<ConcreteFourVector const *>(this); }

    [[nodiscard]] auto impl_lorentz_boost(Vector const &beta, ElementType const &gamma) const -> ConcreteFourVector
    {
        auto const beta2 = dot(beta, beta);
        if (beta2 >= 1)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - abs(beta) is greater than or equal to 1" };

        auto const gamma_minus_one_over_beta2
            = beta2 < 1e-10 ? 0.5 : (gamma - 1) / beta2;
        auto const &F     = *self();
        auto const  gx0   = gamma * ElementType{ F.t };
        auto const  bdotx = dot(beta, F.s);
        return { gx0 + gamma * bdotx, F.s + beta * (gamma_minus_one_over_beta2 * bdotx + gx0) };
    }

    template <long dir>
    [[nodiscard]] auto impl_lorentz_boost(ElementType const &beta, ElementType const &gamma) const -> ConcreteFourVector
    {
        using std::get;
        if (beta * beta < 1) {
            auto const        &F     = *self();
            ConcreteFourVector boost = { gamma * F.t, F.s };
            boost.t += gamma * beta * get<dir>(F);
            get<dir>(boost) = gamma * (get<dir>(F) + beta * *F.t);
            return boost;
        } else {
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - abs(beta) is greater than or equal to 1" };
        }
    }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
