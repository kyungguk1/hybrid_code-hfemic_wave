/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/Predefined.h>
#include <PIC/VT/FourVector.h>
#include <PIC/VT/Vector.h>

LIBPIC_NAMESPACE_BEGIN(1)
struct BorisPush {
    using Vector     = CartVector;
    using FourVector = FourCartVector;

    Real c;         // c
    Real c2;        // c^2
    Real dt_2;      // dt/2
    Real dtOc_2O0;  // (dt/2) * (Oc/O0)
    Real cDtOc_2O0; // c * (dt/2) * (Oc/O0)

    BorisPush(Real const dt, Real const c, Real const O0, Real const Oc) noexcept
    : c{ c }, c2{ c * c }, dt_2{ 0.5 * dt }
    {
        dtOc_2O0  = Oc * dt_2 / O0;
        cDtOc_2O0 = c * dtOc_2O0;
    }

    [[deprecated]] void resistive(Vector &V, Vector B, Vector E, Real nu) const noexcept
    {
        nu *= dt_2;
        B *= dtOc_2O0;
        auto const &cE = E *= cDtOc_2O0;
        //
        // first half acceleration
        //
        V += (cE - nu * V) / (1 + nu / 2);
        //
        // rotation
        //
        V += rotate(V, B);
        //
        // second half acceleration
        //
        V += (cE - nu * V) / (1 + nu / 2);
    }

    /// Non-relativistic Boris push
    ///
    /// @param [in,out] v Particle's velocity
    /// @param B Magnetic field at particle's position
    /// @param E Electric field at particle's position
    ///
    void non_relativistic(Vector &v, Vector B, Vector E) const noexcept
    {
        B *= dtOc_2O0;
        auto const &cE = E *= cDtOc_2O0;
        //
        // first half acceleration
        //
        v += cE;
        //
        // rotation
        //
        v += rotate(v, B);
        //
        // second half acceleration
        //
        v += cE;
    }

    /// Relativistic Boris push
    ///
    /// @param [in,out] gcgv Energy-momentum four-vector, gamma * {c, v}.
    /// @param B Magnetic field at particle's position
    /// @param E Electric field at particle's position
    ///
    void relativistic(FourVector &gcgv, Vector B, Vector E) const noexcept
    {
        B *= dtOc_2O0;
        auto const &cE = E *= cDtOc_2O0;

        // first half acceleration
        gcgv.s += cE;
        gcgv.t = std::sqrt(c2 + dot(gcgv.s, gcgv.s));

        // rotation
        gcgv.s += rotate(gcgv.s, B *= c / *gcgv.t);

        // second half acceleration
        gcgv.s += cE;
        gcgv.t = std::sqrt(c2 + dot(gcgv.s, gcgv.s));
    }

private:
    [[nodiscard]] static Vector rotate(Vector const &v, Vector const &B) noexcept
    {
        return cross(v + cross(v, B), (2 / (1 + dot(B, B))) * B);
    }
};
LIBPIC_NAMESPACE_END(1)
