/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VT/TensorTemplate.h>
#include <PIC/VT/Vector.h>

#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)

// concrete subclass template
#ifndef LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION
#define LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION(ClassName, ConjugateVector)                                 \
    struct ClassName                                                                                        \
    : public Detail::TensorTemplate<ClassName, ConjugateVector> {                                           \
        using ElementType = ConjugateVector::ElementType;                                                   \
        using TensorTemplate::TensorTemplate;                                                               \
        template <class ConcreteTensor, class ConcreteVector>                                               \
        explicit constexpr ClassName(TensorTemplate<ConcreteTensor, ConcreteVector> const &tensor) noexcept \
        : TensorTemplate{ tensor.xx, tensor.yy, tensor.zz, tensor.xy, tensor.yz, tensor.zx }                \
        {                                                                                                   \
        }                                                                                                   \
        static_assert(alignof(TensorTemplate) == alignof(ConjugateVector));                                 \
        static_assert(sizeof(TensorTemplate) == 2 * sizeof(ConjugateVector));                               \
        static_assert(std::is_standard_layout_v<TensorTemplate>);                                           \
    }
#define LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION_WITH_CALCULUS(ClassName, ConjugateVector)                   \
    struct ClassName                                                                                        \
    : public Detail::TensorTemplate<ClassName, ConjugateVector>                                             \
    , public Detail::TensorCalculus<ClassName, ConjugateVector> {                                           \
        using ElementType = ConjugateVector::ElementType;                                                   \
        using TensorTemplate::TensorTemplate;                                                               \
        template <class ConcreteTensor, class ConcreteVector>                                               \
        explicit constexpr ClassName(TensorTemplate<ConcreteTensor, ConcreteVector> const &tensor) noexcept \
        : TensorTemplate{ tensor.xx, tensor.yy, tensor.zz, tensor.xy, tensor.yz, tensor.zx }                \
        {                                                                                                   \
        }                                                                                                   \
        static_assert(alignof(TensorTemplate) == alignof(ConjugateVector));                                 \
        static_assert(sizeof(TensorTemplate) == 2 * sizeof(ConjugateVector));                               \
        static_assert(std::is_standard_layout_v<TensorTemplate>);                                           \
    }
#endif

/// Compact symmetric rank-2 tensor
///
LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION_WITH_CALCULUS(Tensor, Vector);

/// Cartesian tensor
///
LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION_WITH_CALCULUS(CartTensor, CartVector);

/// Field-aligned tensor
///
LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION_WITH_CALCULUS(MFATensor, MFAVector);

/// Covariant & contravariant tensors
///
LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION(CovarTensor, CovarVector);
LIBPIC_STAMP_CONCRETE_TENSOR_DEFINITION(ContrTensor, ContrVector);

[[nodiscard]] inline constexpr auto dot(CovarTensor const &A, ContrVector const &b) noexcept
{
    return CovarVector{ dot(Tensor{ A }, Vector{ b }) };
}
[[nodiscard]] inline constexpr auto dot(ContrVector const &a, CovarTensor const &B) noexcept
{
    return CovarVector{ dot(Vector{ a }, Tensor{ B }) };
}
[[nodiscard]] inline constexpr auto dot(ContrTensor const &A, CovarVector const &b) noexcept
{
    return ContrVector{ dot(Tensor{ A }, Vector{ b }) };
}
[[nodiscard]] inline constexpr auto dot(CovarVector const &a, ContrTensor const &B) noexcept
{
    return ContrVector{ dot(Vector{ a }, Tensor{ B }) };
}
LIBPIC_NAMESPACE_END(1)
