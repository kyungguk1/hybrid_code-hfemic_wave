/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VT/VectorTemplate.h>

#include <complex>

LIBPIC_NAMESPACE_BEGIN(1)
/// Complex vector
///
struct ComplexVector
: public Detail::VectorTemplate<ComplexVector, std::complex<double>>
, public Detail::VectorCalculus<ComplexVector, std::complex<double>> {
    using VectorTemplate::VectorTemplate;
};
using namespace std::literals::complex_literals;
LIBPIC_NAMESPACE_END(1)
