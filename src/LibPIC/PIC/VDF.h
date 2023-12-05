/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/CurviCoord.h>
#include <PIC/Geometry.h>
#include <PIC/Particle.h>
#include <PIC/PlasmaDesc.h>
#include <PIC/Predefined.h>
#include <PIC/UTL/Badge.h>
#include <PIC/UTL/Range.h>
#include <PIC/VT/Scalar.h>
#include <PIC/VT/Tensor.h>
#include <PIC/VT/Vector.h>

#include <vector>

LIBPIC_NAMESPACE_BEGIN(1)
/// Base class for velocity distribution function
///
template <class Concrete>
class VDF {
    using Self = Concrete;

    [[nodiscard]] constexpr decltype(auto) self() const noexcept { return static_cast<Self const *>(this); }
    [[nodiscard]] constexpr decltype(auto) self() noexcept { return static_cast<Self *>(this); }

protected:
    Geometry geomtr;
    Range    domain_extent; // particle boundary extent

    VDF(Geometry const &geo, Range const &domain_extent)
    noexcept
    : geomtr{ geo }, domain_extent{ domain_extent }
    {
    }

public:
    /// Plasma description associated with *this
    ///
    [[nodiscard]] decltype(auto) plasma_desc() const noexcept { return self()->impl_plasma_desc({}); }

    /// Sample a single particle following the marker particle distribution, g0.
    /// \note Concrete subclass should provide impl_emit with the same signature.
    ///
    [[nodiscard]] Particle emit() const { return self()->impl_emit({}); }

    /// Sample N particles following the marker particle distribution, g0.
    /// \note Concrete subclass should provide impl_emit with the same signature.
    [[nodiscard]] std::vector<Particle> emit(unsigned long n) const { return self()->impl_emit({}, n); }

    /// Zero velocity moment at the given position, \<1\>_0(x).
    /// \note Concrete subclass should provide impl_n with the same signature.
    ///
    [[nodiscard]] Scalar n0(CurviCoord const &pos) const { return self()->impl_n({}, pos); }

    /// First velocity moment at the given position, \<v\>_0(x).
    /// \note Concrete subclass should provide impl_nV with the same signature.
    ///
    [[nodiscard]] CartVector nV0(CurviCoord const &pos) const { return self()->impl_nV({}, pos); }

    /// Second velocity moment at the given position, \<vv\>_0(x).
    /// \note Concrete subclass should provide impl_nvv with the same signature.
    ///
    [[nodiscard]] CartTensor nvv0(CurviCoord const &pos) const { return self()->impl_nvv({}, pos); }

    /// Initial physical PSD
    /// \details Concrete subclass should provide impl_f with the same signature.
    [[nodiscard]] Real real_f0(Particle const &ptl) const { return self()->impl_f({}, ptl); }

    /// Ratio of the number of particles at the reference cell to the total number of particles
    /// \note Concrete subclass should provide a member variable, impl_Nrefcell_div_Ntotal, containing this quantity.
    [[nodiscard]] Real Nrefcell_div_Ntotal() const { return self()->impl_Nrefcell_div_Ntotal({}); }
};
LIBPIC_NAMESPACE_END(1)
