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
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
namespace Detail {
/// Symmetric rank-2 four-tensor template
///
template <class ConcreteFourTensor, class Vector, class Tensor>
struct FourTensorTemplate {
    using ElementType = typename Vector::ElementType;

    // tensor elements
    //
    Scalar tt{}; // time component
    Vector ts{}; // mixed components
    Tensor ss{}; // space components

    // constructors
    //
    constexpr FourTensorTemplate() noexcept = default;
    constexpr explicit FourTensorTemplate(ElementType const &v) noexcept
    : tt{ v }, ts{ v }, ss{ v }
    {
    }
    constexpr FourTensorTemplate(Scalar const &tt, Vector const &ts, Tensor const &ss) noexcept
    : tt{ tt }, ts{ ts }, ss{ ss }
    {
    }

    /// Tuple-like get
    ///
    /// \tparam I Row index.
    /// \tparam J Column index.
    /// \param ft FourTensor.
    /// \return Indexed value.
    template <long I, long J>
    [[nodiscard]] constexpr friend auto &get(ConcreteFourTensor const &ft) noexcept
    {
        static_assert(I >= 0 && I < 4, "row index out of range");
        static_assert(J >= 0 && J < 4, "column index out of range");
        return impl_get<I, J>(ft);
    }
    template <long I, long J>
    [[nodiscard]] constexpr friend auto &get(ConcreteFourTensor &ft) noexcept
    {
        static_assert(I >= 0 && I < 4, "row index out of range");
        static_assert(J >= 0 && J < 4, "column index out of range");
        return impl_get<I, J>(ft);
    }

    // compound operations: tensor @= tensor, where @ is one of +, -, *, and /
    // operation is element-wise
    //
    friend constexpr decltype(auto) operator+=(ConcreteFourTensor &lhs, ConcreteFourTensor const &rhs) noexcept
    {
        lhs.tt += rhs.tt;
        lhs.ts += rhs.ts;
        lhs.ss += rhs.ss;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteFourTensor &lhs, ConcreteFourTensor const &rhs) noexcept
    {
        lhs.tt -= rhs.tt;
        lhs.ts -= rhs.ts;
        lhs.ss -= rhs.ss;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteFourTensor &lhs, ConcreteFourTensor const &rhs) noexcept
    {
        lhs.tt *= rhs.tt;
        lhs.ts *= rhs.ts;
        lhs.ss *= rhs.ss;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteFourTensor &lhs, ConcreteFourTensor const &rhs) noexcept
    {
        lhs.tt /= rhs.tt;
        lhs.ts /= rhs.ts;
        lhs.ss /= rhs.ss;
        return lhs;
    }

    // compound operations: tensor @= real, where @ is one of +, -, *, and /
    // operation with scalar is distributed to all elements
    //
    friend constexpr decltype(auto) operator+=(ConcreteFourTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.tt += rhs;
        lhs.ts += rhs;
        lhs.ss += rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator-=(ConcreteFourTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.tt -= rhs;
        lhs.ts -= rhs;
        lhs.ss -= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator*=(ConcreteFourTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.tt *= rhs;
        lhs.ts *= rhs;
        lhs.ss *= rhs;
        return lhs;
    }
    friend constexpr decltype(auto) operator/=(ConcreteFourTensor &lhs, ElementType const &rhs) noexcept
    {
        lhs.tt /= rhs;
        lhs.ts /= rhs;
        lhs.ss /= rhs;
        return lhs;
    }

    // unary operations
    //
    [[nodiscard]] friend constexpr decltype(auto) operator+(ConcreteFourTensor const &F) noexcept { return F; }
    [[nodiscard]] friend constexpr decltype(auto) operator-(ConcreteFourTensor const &F) noexcept { return ConcreteFourTensor{} - F; }

    // binary operations: tensor @ {tensor|real}, where @ is one of +, -, *, and /
    //
    template <class B>
    [[nodiscard]] friend constexpr ConcreteFourTensor operator+(ConcreteFourTensor a, B const &b) noexcept
    {
        a += b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr ConcreteFourTensor operator-(ConcreteFourTensor a, B const &b) noexcept
    {
        a -= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr ConcreteFourTensor operator*(ConcreteFourTensor a, B const &b) noexcept
    {
        a *= b;
        return a;
    }
    template <class B>
    [[nodiscard]] friend constexpr ConcreteFourTensor operator/(ConcreteFourTensor a, B const &b) noexcept
    {
        a /= b;
        return a;
    }

    // binary operations: real @ tensor, where @ is one of +, -, *, and /
    //
    [[nodiscard]] friend constexpr ConcreteFourTensor operator+(ElementType const &b, ConcreteFourTensor const &a) noexcept
    {
        return a + b;
    }
    [[nodiscard]] friend constexpr ConcreteFourTensor operator-(ElementType const &a, ConcreteFourTensor const &b) noexcept
    {
        ConcreteFourTensor A{ a };
        A -= b;
        return A;
    }
    [[nodiscard]] friend constexpr ConcreteFourTensor operator*(ElementType const &b, ConcreteFourTensor const &a) noexcept
    {
        return a * b;
    }
    [[nodiscard]] friend constexpr ConcreteFourTensor operator/(ElementType const &a, ConcreteFourTensor const &b) noexcept
    {
        ConcreteFourTensor A{ a };
        A /= b;
        return A;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, ConcreteFourTensor const &v)
    {
        return os << '{' << v.tt << ", " << v.ts << ", " << v.ss << '}';
    }

private:
    template <long I, long J, class T>
    [[nodiscard]] static constexpr auto &impl_get(T &ft) noexcept
    {
        using std::get;
        if constexpr (I == 0 && J == 0)
            return ft.tt;
        else if constexpr (I > 0 && J == 0)
            return get<I - 1>(ft.ts);
        else if constexpr (I == 0 && J > 0)
            return get<J - 1>(ft.ts);
        else if constexpr (I > 0 && J > 0)
            return get<I - 1, J - 1>(ft.ss);
    }
};

/// Four-tensor calculus interface
///
template <class ConcreteFourTensor, class Vector, class Tensor>
struct FourTensorBoost {
    using ElementType = typename Vector::ElementType;

    /// Minkowski metric with signature (+, -, -, -)
    [[nodiscard]] static constexpr auto minkowski_metric() noexcept
    {
        return ConcreteFourTensor{ 1, {}, -Tensor::identity() };
    }

    /// Lorentz boost
    /// \details Consider the primed frame of reference is moving relative to the lab reference frame
    ///          at a velocity normalized to the light speed, β = v/c. Let n be a unit vector along v.
    ///          The boost matrix, Λ, from the unprimed to primed frame is given by
    ///
    ///              Λ = ( γ          -γβx              -γβy              -γβz      )
    ///                  (-γβx   1 + (γ - 1)nx^2     (γ - 1)nxny       (γ - 1)nxnz  )
    ///                  (-γβy     (γ - 1)nynx     1 + (γ - 1)ny^2     (γ - 1)nynz  )
    ///                  (-γβz     (γ - 1)nznx       (γ - 1)nzny     1 + (γ - 1)nz^2)
    ///
    ///          The inverse transformation is the same as changing the sign of β, i.e., Λ^-1 = Λ(-β).
    ///          The Lorentz boost of four-tensor, F, is given by
    ///
    ///              F^μ'ν' = Λ^μ'_μ Λ^ν'_ν F^μν
    ///
    /// \tparam dir If +1, forward boost, else if -1, inverse boost (same as if β is replaced with -β).
    /// \param F Contravariant four-tensor components.
    /// \param beta β = v/c.
    /// \param gamma Relativistic factor, 1/√(1 - β^2).
    /// \return Contravariant four-tensor components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourTensor lorentz_boost(ConcreteFourTensor const &F, Vector beta, ElementType const &gamma)
    {
        static_assert(dir * dir == 1, "abs(dir) must be 1");
        if constexpr (dir < 0)
            beta *= -1;
        return F.impl_lorentz_boost(beta, gamma);
    }
    /// Lorentz boost
    ///
    /// \tparam dir If +1, forward boost, else if -1, inverse boost (same as if β is replaced with -β).
    /// \param F Contravariant four-tensor components.
    /// \param beta_x_gamma γ times β.
    /// \return Contravariant four-tensor components.
    template <long dir>
    [[nodiscard]] friend ConcreteFourTensor lorentz_boost(ConcreteFourTensor const &F, Vector beta_x_gamma)
    {
        static_assert(dir * dir == 1, "abs(dir) must be 1");
        if constexpr (dir < 0)
            beta_x_gamma *= -1;
        auto const gamma = std::sqrt(1 + dot(beta_x_gamma, beta_x_gamma));
        return F.impl_lorentz_boost(beta_x_gamma / gamma, gamma);
    }

    /// Lorentz boost
    /// \details Consider the primed frame of reference is moving in the x direction relative to the lab reference frame
    ///          at a speed normalized to the light speed, β = v/c.
    ///          The boost matrix, Λ, from the unprimed to primed frame is given by
    ///
    ///              Λ = ( γ   -γβ   0    0)
    ///                  (-γβ   γ    0    0)
    ///                  ( 0    0    1    0)
    ///                  ( 0    0    0    1)
    ///
    ///          The inverse transformation is the same as changing the sign of β, i.e., Λ^-1 = Λ(-β).
    ///          The Lorentz boost of four-tensor, F, is given by
    ///
    ///              F^μ'ν' = Λ^μ'_μ Λ^ν'_ν F^μν
    ///
    /// \tparam dir The direction of boost - 1: x-dir, 2: y-dir, 3: z-dir.
    ///             The negative sign indicates inverse boost.
    /// \param F The four-vector components in the lab frame.
    /// \param beta β = v/c.
    /// \param gamma Relativistic factor, 1/√(1 - β^2).
    /// \return The four-vector components in the moving frame.
    template <long dir>
    [[nodiscard]] friend ConcreteFourTensor lorentz_boost(ConcreteFourTensor const &F, ElementType beta, ElementType const &gamma)
    {
        static_assert(dir * dir >= 1 && dir * dir <= 9, "abs(dir) must be between 1 and 3");
        if constexpr (dir < 0)
            beta *= -1;
        return F.template impl_lorentz_boost<(dir < 0 ? -dir : dir)>(beta, gamma);
    }
    /// Lorentz boost
    ///
    /// \tparam dir The direction of boost - 1: x-dir, 2: y-dir, 3: z-dir.
    ///             The negative sign indicates inverse boost.
    /// \param F The four-vector components in the lab frame.
    /// \param beta_x_gamma γ times β.
    /// \return The four-vector components in the moving frame.
    template <long dir>
    [[nodiscard]] friend ConcreteFourTensor lorentz_boost(ConcreteFourTensor const &F, ElementType beta_x_gamma)
    {
        static_assert(dir * dir >= 1 && dir * dir <= 9, "abs(dir) must be between 1 and 3");
        if constexpr (dir < 0)
            beta_x_gamma *= -1;
        auto const gamma = std::sqrt(1 + beta_x_gamma * beta_x_gamma);
        return F.template impl_lorentz_boost<(dir < 0 ? -dir : dir)>(beta_x_gamma / gamma, gamma);
    }

private:
    [[nodiscard]] auto self() const &noexcept { return static_cast<ConcreteFourTensor const *>(this); }

    /// Lorentz boost matrix
    /// \details Consider the primed frame of reference is moving relative to the lab reference frame
    ///          at a velocity normalized to the light speed, β = v/c. Let n be a unit vector along v.
    ///          The boost matrix, Λ, from the unprimed to primed frame is given by
    ///
    ///              Λ = ( γ          -γβx              -γβy              -γβz      )
    ///                  (-γβx   1 + (γ - 1)nx^2     (γ - 1)nxny       (γ - 1)nxnz  )
    ///                  (-γβy     (γ - 1)nynx     1 + (γ - 1)ny^2     (γ - 1)nynz  )
    ///                  (-γβz     (γ - 1)nznx       (γ - 1)nzny     1 + (γ - 1)nz^2)
    ///
    ///          The inverse transformation is the same as changing the sign of β, i.e., Λ^-1 = Λ(-β).
    ///
    ///          When |β| << 1, the calculation of the unit vector can be problematic. In this case,
    ///          we approximate γ ≈ 1 + β^2/2. Then,
    ///
    ///              Λ ≈ ( γ          -γβx              -γβy              -γβz      )
    ///                  (-γβx     1 + βx^2/2           βxβy/2            βxβz/2    )
    ///                  (-γβy        βyβx/2         1 + βy^2/2           βyβz/2    )
    ///                  (-γβz        βzβx/2            βzβy/2         1 + βz^2/2   )
    ///
    /// \param beta β = v/c.
    /// \param gamma Relativistic factor, 1/√(1 - β^2).
    /// \return Boost matrix.
    [[nodiscard]] auto impl_lorentz_boost(Vector const &beta, ElementType const &gamma) const -> ConcreteFourTensor
    {
        auto const beta2 = dot(beta, beta);
        if (beta2 >= 1)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - abs(beta) is greater than or equal to 1" };

        auto const gamma_minus_one_over_beta2
            = beta2 < 1e-10 ? 0.5 : (gamma - 1) / beta2;
        auto const L = ConcreteFourTensor{
            gamma,
            -gamma * beta,
            {
                1 + gamma_minus_one_over_beta2 * beta.x * beta.x,
                1 + gamma_minus_one_over_beta2 * beta.y * beta.y,
                1 + gamma_minus_one_over_beta2 * beta.z * beta.z,
                0 + gamma_minus_one_over_beta2 * beta.x * beta.y,
                0 + gamma_minus_one_over_beta2 * beta.y * beta.z,
                0 + gamma_minus_one_over_beta2 * beta.z * beta.x,
            }
        };

        constexpr auto pow2 = [](auto const &x) noexcept {
            return x * x;
        };
        auto const &F = *self();
        return {
            Scalar{
                F.ss.xx * pow2(L.ts.x) + 2 * F.ss.xy * L.ts.x * L.ts.y + F.ss.yy * pow2(L.ts.y) + 2 * F.ss.zx * L.ts.x * L.ts.z + 2 * F.ss.yz * L.ts.y * L.ts.z + F.ss.zz * pow2(L.ts.z) + 2 * (F.ts.x * L.ts.x + F.ts.y * L.ts.y + F.ts.z * L.ts.z) * L.tt + F.tt * pow2(L.tt),
            },
            Vector{
                ElementType{ L.ss.xx * (F.ss.xx * L.ts.x + F.ss.xy * L.ts.y + F.ss.zx * L.ts.z + F.ts.x * L.tt) + L.ss.xy * (F.ss.xy * L.ts.x + F.ss.yy * L.ts.y + F.ss.yz * L.ts.z + F.ts.y * L.tt) + L.ss.zx * (F.ss.zx * L.ts.x + F.ss.yz * L.ts.y + F.ss.zz * L.ts.z + F.ts.z * L.tt) + L.ts.x * (F.ts.x * L.ts.x + F.ts.y * L.ts.y + F.ts.z * L.ts.z + F.tt * L.tt) },
                ElementType{ L.ss.xy * (F.ss.xx * L.ts.x + F.ss.xy * L.ts.y + F.ss.zx * L.ts.z + F.ts.x * L.tt) + L.ss.yy * (F.ss.xy * L.ts.x + F.ss.yy * L.ts.y + F.ss.yz * L.ts.z + F.ts.y * L.tt) + L.ss.yz * (F.ss.zx * L.ts.x + F.ss.yz * L.ts.y + F.ss.zz * L.ts.z + F.ts.z * L.tt) + L.ts.y * (F.ts.x * L.ts.x + F.ts.y * L.ts.y + F.ts.z * L.ts.z + F.tt * L.tt) },
                ElementType{ L.ss.zx * (F.ss.xx * L.ts.x + F.ss.xy * L.ts.y + F.ss.zx * L.ts.z + F.ts.x * L.tt) + L.ss.yz * (F.ss.xy * L.ts.x + F.ss.yy * L.ts.y + F.ss.yz * L.ts.z + F.ts.y * L.tt) + L.ss.zz * (F.ss.zx * L.ts.x + F.ss.yz * L.ts.y + F.ss.zz * L.ts.z + F.ts.z * L.tt) + L.ts.z * (F.ts.x * L.ts.x + F.ts.y * L.ts.y + F.ts.z * L.ts.z + F.tt * L.tt) },
            },
            Tensor{
                ElementType{ F.ss.xx * pow2(L.ss.xx) + 2 * F.ss.xy * L.ss.xx * L.ss.xy + F.ss.yy * pow2(L.ss.xy) + 2 * F.ss.zx * L.ss.xx * L.ss.zx + 2 * F.ss.yz * L.ss.xy * L.ss.zx + F.ss.zz * pow2(L.ss.zx) + 2 * (F.ts.x * L.ss.xx + F.ts.y * L.ss.xy + F.ts.z * L.ss.zx) * L.ts.x + F.tt * pow2(L.ts.x) },
                ElementType{ F.ss.xx * pow2(L.ss.xy) + 2 * F.ss.xy * L.ss.xy * L.ss.yy + F.ss.yy * pow2(L.ss.yy) + 2 * F.ss.zx * L.ss.xy * L.ss.yz + 2 * F.ss.yz * L.ss.yy * L.ss.yz + F.ss.zz * pow2(L.ss.yz) + 2 * (F.ts.x * L.ss.xy + F.ts.y * L.ss.yy + F.ts.z * L.ss.yz) * L.ts.y + F.tt * pow2(L.ts.y) },
                ElementType{ F.ss.yy * pow2(L.ss.yz) + 2 * F.ss.xy * L.ss.yz * L.ss.zx + F.ss.xx * pow2(L.ss.zx) + 2 * F.ss.yz * L.ss.yz * L.ss.zz + 2 * F.ss.zx * L.ss.zx * L.ss.zz + F.ss.zz * pow2(L.ss.zz) + 2 * (F.ts.y * L.ss.yz + F.ts.x * L.ss.zx + F.ts.z * L.ss.zz) * L.ts.z + F.tt * pow2(L.ts.z) },
                ElementType{ L.ss.xy * (F.ss.xx * L.ss.xx + F.ss.xy * L.ss.xy + F.ss.zx * L.ss.zx + F.ts.x * L.ts.x) + L.ss.yy * (F.ss.xy * L.ss.xx + F.ss.yy * L.ss.xy + F.ss.yz * L.ss.zx + F.ts.y * L.ts.x) + L.ss.yz * (F.ss.zx * L.ss.xx + F.ss.yz * L.ss.xy + F.ss.zz * L.ss.zx + F.ts.z * L.ts.x) + (F.ts.x * L.ss.xx + F.ts.y * L.ss.xy + F.ts.z * L.ss.zx + F.tt * L.ts.x) * L.ts.y },
                ElementType{ L.ss.zx * (F.ss.xx * L.ss.xy + F.ss.xy * L.ss.yy + F.ss.zx * L.ss.yz + F.ts.x * L.ts.y) + L.ss.yz * (F.ss.xy * L.ss.xy + F.ss.yy * L.ss.yy + F.ss.yz * L.ss.yz + F.ts.y * L.ts.y) + L.ss.zz * (F.ss.zx * L.ss.xy + F.ss.yz * L.ss.yy + F.ss.zz * L.ss.yz + F.ts.z * L.ts.y) + (F.ts.x * L.ss.xy + F.ts.y * L.ss.yy + F.ts.z * L.ss.yz + F.tt * L.ts.y) * L.ts.z },
                ElementType{ L.ss.xx * (F.ss.xy * L.ss.yz + F.ss.xx * L.ss.zx + F.ss.zx * L.ss.zz + F.ts.x * L.ts.z) + L.ss.xy * (F.ss.yy * L.ss.yz + F.ss.xy * L.ss.zx + F.ss.yz * L.ss.zz + F.ts.y * L.ts.z) + L.ss.zx * (F.ss.yz * L.ss.yz + F.ss.zx * L.ss.zx + F.ss.zz * L.ss.zz + F.ts.z * L.ts.z) + L.ts.x * (F.ts.y * L.ss.yz + F.ts.x * L.ss.zx + F.ts.z * L.ss.zz + F.tt * L.ts.z) },
            }
        };
    }
    template <long dir>
    [[nodiscard]] auto impl_lorentz_boost(ElementType const &beta, ElementType const &gamma) const -> ConcreteFourTensor
    {
        if (beta * beta >= 1)
            throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - abs(beta) is greater than or equal to 1" };

        constexpr auto pow2 = [](auto const &x) noexcept {
            return x * x;
        };
        auto &F = *self();
        if constexpr (dir == 1) {
            return {
                Scalar{
                    (F.tt + beta * (-2 * F.ts.x + beta * F.ss.xx)) * pow2(gamma),
                },
                Vector{
                    (F.ts.x + pow2(beta) * F.ts.x - beta * (*F.tt + F.ss.xx)) * pow2(gamma),
                    (F.ts.y - beta * F.ss.xy) * gamma,
                    (F.ts.z - beta * F.ss.zx) * gamma,
                },
                Tensor{
                    (pow2(beta) * *F.tt - 2 * beta * F.ts.x + F.ss.xx) * pow2(gamma),
                    F.ss.yy,
                    F.ss.zz,
                    (-(beta * F.ts.y) + F.ss.xy) * gamma,
                    F.ss.yz,
                    (-(beta * F.ts.z) + F.ss.zx) * gamma,
                },
            };
        } else if constexpr (dir == 2) {
            return {
                Scalar{
                    (F.tt + beta * (-2 * F.ts.y + beta * F.ss.yy)) * pow2(gamma),
                },
                Vector{
                    (F.ts.x - beta * F.ss.xy) * gamma,
                    (F.ts.y + pow2(beta) * F.ts.y - beta * (*F.tt + F.ss.yy)) * pow2(gamma),
                    (F.ts.z - beta * F.ss.yz) * gamma,
                },
                Tensor{
                    F.ss.xx,
                    (pow2(beta) * *F.tt - 2 * beta * F.ts.y + F.ss.yy) * pow2(gamma),
                    F.ss.zz,
                    (-(beta * F.ts.x) + F.ss.xy) * gamma,
                    (-(beta * F.ts.z) + F.ss.yz) * gamma,
                    F.ss.zx,
                },
            };
        } else if constexpr (dir == 3) {
            return {
                Scalar{
                    (F.tt + beta * (-2 * F.ts.z + beta * F.ss.zz)) * pow2(gamma),
                },
                Vector{
                    (F.ts.x - beta * F.ss.zx) * gamma,
                    (F.ts.y - beta * F.ss.yz) * gamma,
                    (F.ts.z + pow2(beta) * F.ts.z - beta * (*F.tt + F.ss.zz)) * pow2(gamma),
                },
                Tensor{
                    F.ss.xx,
                    F.ss.yy,
                    (pow2(beta) * *F.tt - 2 * beta * F.ts.z + F.ss.zz) * pow2(gamma),
                    F.ss.xy,
                    (-(beta * F.ts.y) + F.ss.yz) * gamma,
                    (-(beta * F.ts.x) + F.ss.zx) * gamma,
                },
            };
        }
    }
};
} // namespace Detail
LIBPIC_NAMESPACE_END(1)
