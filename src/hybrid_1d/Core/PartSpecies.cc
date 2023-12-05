/*
 * Copyright (c) 2019-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "PartSpecies.h"
#include "BField.h"
#include "EField.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

HYBRID1D_BEGIN_NAMESPACE
namespace {
template <class T, long N>
decltype(auto) operator*=(GridArray<T, N, Pad> &G, T const &w) noexcept
{ // include padding
    G.for_all([&w](T &value_ref) {
        value_ref *= w;
    });
    return G;
}

auto const &half_grid(Grid<CartVector> &full, BField const &half) noexcept
{
    for (long i = -Pad; i < full.size() - 1 + Pad; ++i) {
        (full[i] = half[i + 1] + half[i + 0]) *= 0.5;
    }
    full.dead_end()[-1] = CartVector{ std::numeric_limits<Real>::quiet_NaN() };
    return full;
}
} // namespace

// ctor
//
PartSpecies::PartSpecies(ParamSet const &params, KineticPlasmaDesc const &_desc, std::unique_ptr<VDFVariant> _vdf)
: Species{ params }
, desc{ _desc }
, vdf{ std::move(_vdf) }
, Nc{ Particle::quiet_nan }
{
    // calculate the number of particles at the center cell
    if (long const Np = params.Nx * desc.Nc)
        Nc = Np * vdf->Nrefcell_div_Ntotal();
    else
        Nc = 1;

    switch (desc.shape_order) {
        case ShapeOrder::_1st:
            // Pad > 1 if 1st-order shape
            if constexpr (Pad > 1) {
                m_update_velocity = &PartSpecies::impl_update_velocity<1>;
                m_collect_part    = &PartSpecies::impl_collect_part<1>;
            }
            break;
        case ShapeOrder::_2nd:
            // Pad > 2 if 2nd-order shape
            if constexpr (Pad > 2) {
                m_update_velocity = &PartSpecies::impl_update_velocity<2>;
                m_collect_part    = &PartSpecies::impl_collect_part<2>;
            }
            break;
        case ShapeOrder::_3rd:
            // Pad >= 3 if 3rd-order shape
            if constexpr (Pad >= 3) {
                m_update_velocity = &PartSpecies::impl_update_velocity<3>;
                m_collect_part    = &PartSpecies::impl_collect_part<3>;
            }
            break;
    }
}
void PartSpecies::populate(long const color, long const divisor)
{
    if (divisor <= 0)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - non-positive divisor" };
    if (color < 0 || color >= divisor)
        throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ } + " - invalid color range" };

    // cache the equilibrium moments
    auto const q1min = grid_subdomain_extent().min();
    for (long i = 0; i < equilibrium_mom0.size(); ++i) { // only the interior
        CurviCoord const pos{ i + q1min };
        equilibrium_mom0[i] = vdf->n0(pos);
        equilibrium_mom1[i] = vdf->nV0(pos);
        equilibrium_mom2[i] = vdf->nvv0(pos);
    }

    // populate particles
    long       Np_within_this_subdomain = 0;
    auto const pred                     = [&](auto const &) {
        // must increment the count only if all other tests are passed
        return color == Np_within_this_subdomain++ % divisor;
    };
    bucket.clear();
    for (long i = 0; i < params.Nx; ++i) {
        load_ptls(vdf->emit(static_cast<unsigned long>(desc.Nc)), pred);
    }
}

// update & collect interface
//
void PartSpecies::update_vel(BField const &bfield, EField const &efield, Real const dt)
{
    (this->*m_update_velocity)(bucket, half_grid(moment<1>(), bfield), efield, BorisPush{ dt, params.c, params.O0, desc.Oc });
    impl_update_weight(bucket, desc.psd_refresh_frequency * dt);
}
void PartSpecies::update_pos(Real const dt, Real const fraction_of_grid_size_allowed_to_travel)
{
    if (!impl_update_pos(bucket, dt, 1.0 / fraction_of_grid_size_allowed_to_travel))
        throw std::domain_error{ std::string{ __PRETTY_FUNCTION__ } + " - particle(s) moved too far" };
}
void PartSpecies::collect_part()
{
    // collect moments
    (this->*m_collect_part)(moment<0>(), moment<1>());

    // add equilibrium moments
    if (desc.scheme) {
        auto const collect = [w = m_moment_weighting_factor](auto const &equilibrium, auto const &delta) {
            return delta + equilibrium * w;
        };
        std::transform(equilibrium_mom0.dead_begin(), equilibrium_mom0.dead_end(), moment<0>().dead_begin(), moment<0>().dead_begin(), collect);
        std::transform(equilibrium_mom1.dead_begin(), equilibrium_mom1.dead_end(), moment<1>().dead_begin(), moment<1>().dead_begin(), collect);
    }
}
void PartSpecies::collect_all()
{
    // collect moments
    impl_collect_all(moment<0>(), moment<1>(), moment<2>());

    // add equilibrium moments
    if (desc.scheme) {
        auto const collect = [w = m_moment_weighting_factor](auto const &equilibrium, auto const &delta) {
            return delta + equilibrium * w;
        };
        std::transform(equilibrium_mom0.dead_begin(), equilibrium_mom0.dead_end(), moment<0>().dead_begin(), moment<0>().dead_begin(), collect);
        std::transform(equilibrium_mom1.dead_begin(), equilibrium_mom1.dead_end(), moment<1>().dead_begin(), moment<1>().dead_begin(), collect);
        std::transform(equilibrium_mom2.dead_begin(), equilibrium_mom2.dead_end(), moment<2>().dead_begin(), moment<2>().dead_begin(), collect);
    }
}

// heavy lifting
//
bool PartSpecies::impl_update_pos(bucket_type &bucket, Real const dt, Real const travel_distance_scale_factor) const
{
    bool did_not_move_too_far = true;
    for (auto &ptl : bucket) {
        auto moved = CurviCoord{ geomtr.cart_to_contr(ptl.vel, ptl.pos) } * dt;
        ptl.pos += moved;

        // travel distance check
        moved *= travel_distance_scale_factor;
        did_not_move_too_far &= 0 == long(moved.q1);
    }
    return did_not_move_too_far;
}

template <long Order>
void PartSpecies::impl_update_velocity(bucket_type &bucket, Grid<CartVector> const &dB, EField const &E, BorisPush const &boris) const
{
    static_assert(Pad >= Order, "shape order should be less than or equal to the number of ghost cells");
    auto const q1min = grid_subdomain_extent().min();
    for (auto &ptl : bucket) {
        Shape<Order> const sx{ ptl.pos.q1 - q1min };

        // get gyro-radius offset: rL = e1 x v / Oc (Oc is signed)
        auto const Oc   = desc.Oc * geomtr.Bmag_div_B0(ptl.pos);
        auto const rL_y = -ptl.vel.z / Oc;
        auto const rL_z = +ptl.vel.y / Oc;

        boris.non_relativistic(ptl.vel, geomtr.Bcart(ptl.pos, rL_y, rL_z) + dB.interp(sx), E.interp(sx));
    }
}

void PartSpecies::impl_update_weight(bucket_type &bucket, Real const nu_dt) const
{
    auto const &vdf = *this->vdf;

    // the weight is given by
    //
    //     f(t, x(t), v(t))/g(0, x(0), v(0)) - δ*f_0(x(t), v(t))/g(0, x(0), v(0))
    //
    // where g is the marker particle distribution and δ is 0 for full-f and 1 for delta-f.
    //
    switch (desc.scheme) {
        case ParticleScheme::full_f: {
            if (desc.should_refresh_psd) {
                for (auto &ptl : bucket) {
                    ptl.psd.real_f = (ptl.psd.real_f + nu_dt * vdf.real_f0(ptl)) / (1 + nu_dt);
                    ptl.psd.weight = ptl.psd.real_f / ptl.psd.marker;
                }
            }
            break;
        }
        case ParticleScheme::delta_f: {
            for (auto &ptl : bucket) {
                auto const f0 = vdf.real_f0(ptl);
                if (desc.should_refresh_psd)
                    ptl.psd.real_f = (ptl.psd.real_f + nu_dt * f0) / (1 + nu_dt);
                ptl.psd.weight = (ptl.psd.real_f - f0) / ptl.psd.marker;
            }
            break;
        }
    }
}

template <long Order>
void PartSpecies::impl_collect_part(Grid<Scalar> &n, Grid<CartVector> &nV) const
{
    static_assert(Pad >= Order, "shape order should be less than or equal to the number of ghost cells");
    auto const q1min = grid_subdomain_extent().min();
    n.fill_all(Scalar{});
    nV.fill_all(CartVector{});
    for (auto const &ptl : bucket) {
        Shape<Order> const sx{ ptl.pos.q1 - q1min };
        n.deposit(sx, ptl.psd.weight);
        nV.deposit(sx, ptl.vel * ptl.psd.weight);
    }
    n *= Scalar{ 1.0 / Nc };
    nV *= CartVector{ 1.0 / Nc };
}
void PartSpecies::impl_collect_all(Grid<Scalar> &n, Grid<CartVector> &nV, Grid<CartTensor> &nvv) const
{
    n.fill_all(Scalar{});
    nV.fill_all(CartVector{});
    nvv.fill_all(CartTensor{});
    CartTensor tmp{};
    auto const q1min = grid_subdomain_extent().min();
    for (auto const &ptl : bucket) {
        Shape<1> const sx{ ptl.pos.q1 - q1min };
        n.deposit(sx, ptl.psd.weight);
        nV.deposit(sx, ptl.vel * ptl.psd.weight);
        tmp.hi() = tmp.lo() = ptl.vel;
        tmp.lo() *= ptl.vel;                             // diagonal part; {vx*vx, vy*vy, vz*vz}
        tmp.hi() *= { ptl.vel.y, ptl.vel.z, ptl.vel.x }; // off-diag part; {vx*vy, vy*vz, vz*vx}
        nvv.deposit(sx, tmp *= ptl.psd.weight);
    }
    n *= Scalar{ 1.0 / Nc };
    nV *= CartVector{ 1.0 / Nc };
    nvv *= CartTensor{ 1.0 / Nc };
}

template <class Object>
auto write_attr(Object &obj, PartSpecies const &sp) -> decltype(obj)
{
    obj.attribute("Nc", hdf5::make_type(sp->Nc), hdf5::Space::scalar())
        .write(sp->Nc);
    obj.attribute("Nrefcell_div_Ntotal", hdf5::make_type(sp.vdf->Nrefcell_div_Ntotal()), hdf5::Space::scalar())
        .write(sp.vdf->Nrefcell_div_Ntotal());
    obj.attribute("shape_order", hdf5::make_type<long>(sp->shape_order), hdf5::Space::scalar())
        .template write<long>(sp->shape_order);
    obj.attribute("particle_scheme", hdf5::make_type<long>(sp->scheme), hdf5::Space::scalar())
        .template write<long>(sp->scheme);
    obj.attribute("initial_weight", hdf5::make_type(sp->initial_weight), hdf5::Space::scalar())
        .write(sp->initial_weight);
    obj.attribute("marker_temp_ratio", hdf5::make_type(sp->marker_temp_ratio), hdf5::Space::scalar())
        .write(sp->marker_temp_ratio);
    obj.attribute("psd_refresh_frequency", hdf5::make_type(sp->psd_refresh_frequency), hdf5::Space::scalar())
        .write(sp->psd_refresh_frequency);

    return obj << static_cast<Species const &>(sp);
}
auto operator<<(hdf5::Group &obj, PartSpecies const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
auto operator<<(hdf5::Dataset &obj, PartSpecies const &sp) -> decltype(obj)
{
    return write_attr(obj, sp);
}
HYBRID1D_END_NAMESPACE
