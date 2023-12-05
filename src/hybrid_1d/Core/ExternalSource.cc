/*
 * Copyright (c) 2022-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "ExternalSource.h"
#include <PIC/Shape.h>
#include <complex>

HYBRID1D_BEGIN_NAMESPACE
void ExternalSource::update(Real const delta_t)
{
    auto const t = Real(m_cur_step++) * params.dt + delta_t;
    collect(moment<0>(), moment<1>(), t);
}

void ExternalSource::collect(Grid<Scalar> &rho, Grid<CartVector> &J, Real const t) const
{
    rho.fill_all(Scalar{});
    J.fill_all(CartVector{});
    auto const domain_extent = grid_subdomain_extent();
    auto const q1min         = domain_extent.min();
    for (unsigned i = 0; i < number_of_source_points; ++i) {
        if (auto const pos = src_pos.at(i); domain_extent.is_member(pos.q1)) {
            Shape<1> const sx{ pos.q1 - q1min };
            auto const     J_mfa = current(src_Jre.at(i), src_Jim.at(i), t);
            J.deposit(sx, geomtr.mfa_to_cart(J_mfa, pos) * (envelope(t) * m_moment_weighting_factor));
        }
    }
}
auto ExternalSource::current(MFAVector const &J0re, MFAVector const &J0im, Real const t) const noexcept -> MFAVector
{
    Real const phase = -src_desc.omega * (t - src_desc.extent.loc);
    using namespace std::literals::complex_literals;
    auto const exp = std::exp(1i * phase);
    using Cx       = std::complex<Real>;
    return {
        (Cx{ J0re.x, J0im.x } * exp).real(),
        (Cx{ J0re.y, J0im.y } * exp).real(),
        (Cx{ J0re.z, J0im.z } * exp).real(),
    };
}
auto ExternalSource::envelope(Real t) const noexcept -> Real
{
    t -= src_desc.extent.loc; // change the origin

    // before ease-in
    if (t < -src_desc.ease.in)
        return 0;

    // ease-in phase
    if (t < 0)
        return .5 * (1 + std::cos(ramp_slope.ease_in * t));

    // middle phase
    if (t < src_desc.extent.len)
        return 1;

    // ease-out phase
    if ((t -= src_desc.extent.len) < src_desc.ease.out)
        return .5 * (1 + std::cos(ramp_slope.ease_out * t));

    // after ease-out
    return 0;
}

template <class Object>
auto write_attr(Object &obj, ExternalSource const &sp) -> decltype(obj)
{
    using hdf5::make_type;
    using hdf5::Space;

    obj.attribute("source_omega", make_type(sp.src_desc.omega), Space::scalar())
        .write(sp.src_desc.omega);
    obj.attribute("source_start", make_type(sp.src_desc.extent.loc), Space::scalar())
        .write(sp.src_desc.extent.loc);
    obj.attribute("source_duration", make_type(sp.src_desc.extent.len), Space::scalar())
        .write(sp.src_desc.extent.len);
    obj.attribute("source_ease_in", make_type(sp.src_desc.ease.in), Space::scalar())
        .write(sp.src_desc.ease.in);
    obj.attribute("source_ease_out", make_type(sp.src_desc.ease.out), Space::scalar())
        .write(sp.src_desc.ease.out);
    obj.attribute("source_position", make_type<Real>(), Space::simple(sp.number_of_source_points))
        .write(sp.src_pos.data(), make_type<Real>());
    obj.attribute("source_J0re", make_type<Real>(), Space::simple({ sp.number_of_source_points, 3 }))
        .write(sp.src_Jre.data(), make_type<Real>());
    obj.attribute("source_J0im", make_type<Real>(), Space::simple({ sp.number_of_source_points, 3 }))
        .write(sp.src_Jim.data(), make_type<Real>());

    return obj << static_cast<Species const &>(sp);
}
auto operator<<(hdf5::Group &obj, ExternalSource const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
auto operator<<(hdf5::Dataset &obj, ExternalSource const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
HYBRID1D_END_NAMESPACE
