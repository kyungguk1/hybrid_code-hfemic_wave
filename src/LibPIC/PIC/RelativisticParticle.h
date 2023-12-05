/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/CurviCoord.h>
#include <PIC/VT/FourVector.h>
#include <PIC/VT/Vector.h>

#include <limits>
#include <ostream>
#include <type_traits>

LIBPIC_NAMESPACE_BEGIN(1)
/// single relativistic particle
///
struct RelativisticParticle {
    using Vector                    = CartVector;
    using FourVector                = FourCartVector;
    using Real                      = FourVector::ElementType;
    static constexpr auto quiet_nan = std::numeric_limits<Real>::quiet_NaN();

    // for delta-f
    struct PSD {
        Real weight{ quiet_nan }; // f(0, x(0), v(0))/g(0, x(0), v(0)) - f_0(x(t), v(t))/g(0, x(0), v(0))
        Real real_f{ quiet_nan }; // f(0, x(0), v(0)), physical particle PSD
        Real marker{ quiet_nan }; // g(0, x(0), v(0)), simulation particle PSD

        constexpr PSD() noexcept = default;
        constexpr PSD(Real w, Real f, Real g) noexcept
        : weight{ w }, real_f{ f }, marker{ g } {}
    };

    FourVector gcgvel{ quiet_nan }; //!< γ*{c, v}
    CurviCoord pos{ quiet_nan };    //!< curvilinear coordinates
    PSD        psd{};
    long       id{ -1 }; //!< particle identifier

    [[nodiscard]] auto beta() const noexcept
    {
        return gcgvel.s / Real{ gcgvel.t };
    }
    [[nodiscard]] auto velocity(Real const c) const noexcept
    {
        return gcgvel.s * (c / *gcgvel.t); // usual velocity
    }

    RelativisticParticle() noexcept = default;
    RelativisticParticle(FourVector const &gcgvel, CurviCoord const &pos) noexcept
    : gcgvel{ gcgvel }, pos{ pos }, id{ next_id() }
    {
    }

private:
    [[nodiscard]] static long next_id() noexcept
    {
        thread_local static long next_id{ 0 };
        return next_id++;
    }

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, RelativisticParticle const &ptl)
    {
        return os << '{' << ptl.gcgvel.s << ", " << ptl.pos << ", "
                  << '{' << ptl.psd.weight << ", " << ptl.psd.real_f << ", " << ptl.psd.marker << '}' << ", "
                  << ptl.gcgvel.t << ", " << ptl.id << '}';
    }
};
static_assert(sizeof(RelativisticParticle) == 9 * sizeof(RelativisticParticle::Real));
static_assert(alignof(RelativisticParticle) == alignof(RelativisticParticle::FourVector));
static_assert(std::is_standard_layout_v<RelativisticParticle>);
LIBPIC_NAMESPACE_END(1)
