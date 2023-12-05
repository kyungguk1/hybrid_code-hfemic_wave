/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/VT/FourVectorTemplate.h>
#include <PIC/VT/Scalar.h>
#include <PIC/VT/Vector.h>

LIBPIC_NAMESPACE_BEGIN(1)
// forward decls
template <class... Args>
auto lorentz_boost(Args &&...);

// concrete subclass template
#ifndef LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION
#define LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION(ClassName, ConjugateVector)                                \
    struct ClassName                                                                                            \
    : public Detail::FourVectorTemplate<ClassName, ConjugateVector> {                                           \
        using ElementType = ConjugateVector::ElementType;                                                       \
        using FourVectorTemplate::FourVectorTemplate;                                                           \
        template <class ConcreteFourVector, class ConcreteVector>                                               \
        explicit constexpr ClassName(FourVectorTemplate<ConcreteFourVector, ConcreteVector> const &FV) noexcept \
        : FourVectorTemplate{ FV.t, ConjugateVector{ FV.s } }                                                   \
        {                                                                                                       \
        }                                                                                                       \
    }
#define LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION_WITH_BOOST(ClassName, ConjugateVector)                     \
    struct ClassName                                                                                            \
    : public Detail::FourVectorTemplate<ClassName, ConjugateVector>                                             \
    , public Detail::FourVectorBoost<ClassName, ConjugateVector> {                                              \
        using ElementType = ConjugateVector::ElementType;                                                       \
        using FourVectorTemplate::FourVectorTemplate;                                                           \
        template <class ConcreteFourVector, class ConcreteVector>                                               \
        explicit constexpr ClassName(FourVectorTemplate<ConcreteFourVector, ConcreteVector> const &FV) noexcept \
        : FourVectorTemplate{ FV.t, ConjugateVector{ FV.s } }                                                   \
        {                                                                                                       \
        }                                                                                                       \
    }
#endif

/// Four-vector
///
LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION_WITH_BOOST(FourVector, Vector);

/// Cartesian four-vector
///
LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION_WITH_BOOST(FourCartVector, CartVector);

/// Field-aligned four-vector
///
LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION_WITH_BOOST(FourMFAVector, MFAVector);

/// Covariant & contravariant four-vectors
///
LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION(FourCovarVector, CovarVector);
LIBPIC_STAMP_CONCRETE_FOUR_VECTOR_DEFINITION(FourContrVector, ContrVector);

[[nodiscard]] inline constexpr auto dot(FourCovarVector const &A, FourContrVector const &B) noexcept
{
    return *(A.t * B.t) + dot(A.s, B.s);
}
[[nodiscard]] inline constexpr auto dot(FourContrVector const &A, FourCovarVector const &B) noexcept
{
    return *(A.t * B.t) + dot(A.s, B.s);
}
LIBPIC_NAMESPACE_END(1)
