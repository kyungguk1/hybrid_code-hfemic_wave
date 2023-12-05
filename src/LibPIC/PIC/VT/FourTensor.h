/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/VT/FourTensorTemplate.h>
#include <PIC/VT/FourVector.h>
#include <PIC/VT/Scalar.h>
#include <PIC/VT/Tensor.h>
#include <PIC/VT/Vector.h>

#include <cmath>
#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
// forward decls
template <class... Args>
auto lorentz_boost(Args &&...);

// concrete subclass template
#ifndef LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION
#define LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION(ClassName, ConjugateVector, ConjugateTensor)                               \
    struct ClassName                                                                                                            \
    : public Detail::FourTensorTemplate<ClassName, ConjugateVector, ConjugateTensor> {                                          \
        using ElementType = ConjugateVector::ElementType;                                                                       \
        using FourTensorTemplate::FourTensorTemplate;                                                                           \
        template <class ConcreteFourTensor, class ConcreteVector, class ConcreteTensor>                                         \
        explicit constexpr ClassName(FourTensorTemplate<ConcreteFourTensor, ConcreteVector, ConcreteTensor> const &FT) noexcept \
        : FourTensorTemplate{ FT.tt, ConjugateVector{ FT.ts }, ConjugateTensor{ FT.ss } }                                       \
        {                                                                                                                       \
        }                                                                                                                       \
    }
#define LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION_WITH_BOOST(ClassName, ConjugateVector, ConjugateTensor)                    \
    struct ClassName                                                                                                            \
    : public Detail::FourTensorTemplate<ClassName, ConjugateVector, ConjugateTensor>                                            \
    , public Detail::FourTensorBoost<ClassName, ConjugateVector, ConjugateTensor> {                                             \
        using ElementType = ConjugateVector::ElementType;                                                                       \
        using FourTensorTemplate::FourTensorTemplate;                                                                           \
        template <class ConcreteFourTensor, class ConcreteVector, class ConcreteTensor>                                         \
        explicit constexpr ClassName(FourTensorTemplate<ConcreteFourTensor, ConcreteVector, ConcreteTensor> const &FT) noexcept \
        : FourTensorTemplate{ FT.tt, ConjugateVector{ FT.ts }, ConjugateTensor{ FT.ss } }                                       \
        {                                                                                                                       \
        }                                                                                                                       \
    }
#endif

/// Symmetric rank-2 four-tensor
///
LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION_WITH_BOOST(FourTensor, Vector, Tensor);

/// Cartesian four-tensor
///
LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION_WITH_BOOST(FourCartTensor, CartVector, CartTensor);

/// Field-aligned four-tensor
///
LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION_WITH_BOOST(FourMFATensor, MFAVector, MFATensor);

/// Covariant & contravariant four-tensors
///
LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION(FourCovarTensor, CovarVector, CovarTensor);
LIBPIC_STAMP_CONCRETE_FOUR_TENSOR_DEFINITION(FourContrTensor, ContrVector, ContrTensor);

[[nodiscard]] inline constexpr auto dot(FourContrTensor const &A, FourCovarVector const &V) noexcept -> FourContrVector
{
    return { A.tt * V.t + dot(A.ts, V.s), A.ts * *V.t + dot(A.ss, V.s) };
}
[[nodiscard]] inline constexpr auto dot(FourCovarVector const &V, FourContrTensor const &A) noexcept -> FourContrVector
{
    return dot(A, V); // because A^T == A
}
[[nodiscard]] inline constexpr auto dot(FourCovarTensor const &A, FourContrVector const &V) noexcept -> FourCovarVector
{
    return { A.tt * V.t + dot(A.ts, V.s), A.ts * *V.t + dot(A.ss, V.s) };
}
[[nodiscard]] inline constexpr auto dot(FourContrVector const &V, FourCovarTensor const &A) noexcept -> FourCovarVector
{
    return dot(A, V); // because A^T == A
}
LIBPIC_NAMESPACE_END(1)
