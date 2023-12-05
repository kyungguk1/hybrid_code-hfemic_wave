/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VT/VectorTemplate.h>

#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)

// concrete subclass template
#ifndef LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION
#define LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION(ClassName)                                               \
    struct ClassName                                                                                     \
    : public Detail::VectorTemplate<ClassName, double> {                                                 \
        using ElementType = double;                                                                      \
        using VectorTemplate::VectorTemplate;                                                            \
        template <class ConcreteVector>                                                                  \
        explicit constexpr ClassName(VectorTemplate<ConcreteVector, ElementType> const &vector) noexcept \
        : VectorTemplate{ vector.x, vector.y, vector.z }                                                 \
        {                                                                                                \
        }                                                                                                \
        static_assert(24 == sizeof(VectorTemplate));                                                     \
        static_assert(8 == alignof(VectorTemplate));                                                     \
        static_assert(std::is_standard_layout_v<VectorTemplate>);                                        \
    }
#define LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION_WITH_CALCULUS(ClassName)                                 \
    struct ClassName                                                                                     \
    : public Detail::VectorTemplate<ClassName, double>                                                   \
    , public Detail::VectorCalculus<ClassName, double> {                                                 \
        using ElementType = double;                                                                      \
        using VectorTemplate::VectorTemplate;                                                            \
        template <class ConcreteVector>                                                                  \
        explicit constexpr ClassName(VectorTemplate<ConcreteVector, ElementType> const &vector) noexcept \
        : VectorTemplate{ vector.x, vector.y, vector.z }                                                 \
        {                                                                                                \
        }                                                                                                \
        static_assert(24 == sizeof(VectorTemplate));                                                     \
        static_assert(8 == alignof(VectorTemplate));                                                     \
        static_assert(std::is_standard_layout_v<VectorTemplate>);                                        \
    }
#endif

/// Real vector
///
LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION_WITH_CALCULUS(Vector);

/// Cartesian vector
///
LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION_WITH_CALCULUS(CartVector);

/// Field-aligned vector
///
LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION_WITH_CALCULUS(MFAVector);

/// Covariant & contravariant vectors
///
LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION(CovarVector);
LIBPIC_STAMP_CONCRETE_VECTOR_DEFINITION(ContrVector);

[[nodiscard]] inline constexpr auto dot(CovarVector const &A, ContrVector const &B) noexcept
{
    return dot(Vector{ A }, Vector{ B });
}
[[nodiscard]] inline constexpr auto dot(ContrVector const &A, CovarVector const &B) noexcept
{
    return dot(Vector{ A }, Vector{ B });
}
[[nodiscard]] inline constexpr auto cross(CovarVector const &A, CovarVector const &B, double const sqrt_g) noexcept
{
    return ContrVector{ cross(Vector{ A }, Vector{ B }) / sqrt_g };
}
[[nodiscard]] inline constexpr auto cross(ContrVector const &A, ContrVector const &B, double const sqrt_g) noexcept
{
    return CovarVector{ cross(Vector{ A }, Vector{ B }) * sqrt_g };
}
LIBPIC_NAMESPACE_END(1)
