/*
 * Copyright (c) 2019-2023, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/CurviCoord.h>
#include <PIC/Predefined.h>
#include <PIC/UTL/Range.h>
#include <PIC/VT/ComplexVector.h>
#include <PIC/VT/Vector.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

LIBPIC_NAMESPACE_BEGIN(1)
/// Common parameters for all plasma populations
///
struct PlasmaDesc {
    Real Oc;                          //!< Cyclotron frequency.
    Real op;                          //!< Plasma frequency.
    long number_of_source_smoothings; //!< The number of source smoothings.

    /// Construct a plasma description
    /// \param Oc Cyclotron frequency of this plasma species/component.
    /// \param op Plasma frequency of this plasma species/component.
    /// \param n_smooths An optional argument to set the number of source smoothing. Default is 0.
    /// \throw Throws std::invalid_argument if either Oc == 0 or op <= 0.
    ///
    constexpr PlasmaDesc(Real Oc, Real op, unsigned n_smooths = {})
    : Oc{ Oc }, op{ op }, number_of_source_smoothings{ n_smooths }
    {
        if (this->Oc == 0)
            throw std::invalid_argument{ "Oc should not be zero" };
        if (this->op <= 0)
            throw std::invalid_argument{ "op should be positive" };
    }

protected:
    PlasmaDesc() noexcept = default;

    static constexpr auto quiet_nan = std::numeric_limits<Real>::quiet_NaN();
    constexpr explicit PlasmaDesc(unsigned n_smooths) noexcept
    : Oc{ quiet_nan }, op{ quiet_nan }, number_of_source_smoothings{ n_smooths }
    {
    }

    [[nodiscard]] friend constexpr auto serialize(PlasmaDesc const &desc) noexcept
    {
        return std::make_tuple(desc.Oc, desc.op);
    }
    [[nodiscard]] friend constexpr bool operator==(PlasmaDesc const &lhs, PlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Charge-neutralizing electron fluid descriptor.
///
struct eFluidDesc : public PlasmaDesc {
    Real beta;  //!< Electron beta.
    Real gamma; //!< Specific heat ratio, gamma.

    /// Construct charge-neutralizing, massless electron fluid description
    /// \param desc Common plasma description.
    /// \param beta Electron plasma beta. Default is 0.
    /// \param closure Polytropic index in equation of state. Default is adiabatic.
    explicit constexpr eFluidDesc(PlasmaDesc const &desc, Real beta = {}, Closure closure = adiabatic)
    : PlasmaDesc(desc), beta{ beta }, gamma(closure / 10)
    {
        gamma /= closure % 10;
        if (this->beta < 0)
            throw std::invalid_argument{ "beta should be non-negative" };
    }

private:
    [[nodiscard]] friend constexpr auto serialize(eFluidDesc const &desc) noexcept
    {
        PlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(desc.beta, desc.gamma));
    }
};

/// Parameter set for a cold plasma population
///
struct ColdPlasmaDesc : public PlasmaDesc {
    // the explicit qualifier is to prevent an accidental construction of an empty object
    //
    explicit ColdPlasmaDesc() noexcept = default;

    /// Construct a cold plasma description
    /// \param desc Common plasma description.
    /// \throw Any exception thrown by PlasmaDesc.
    ///
    explicit constexpr ColdPlasmaDesc(PlasmaDesc const &desc)
    : PlasmaDesc(desc) {}

private:
    [[nodiscard]] friend constexpr auto serialize(ColdPlasmaDesc const &desc) noexcept
    {
        PlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base));
    }
    [[nodiscard]] friend constexpr bool operator==(ColdPlasmaDesc const &lhs, ColdPlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Common parameters for all kinetic plasma populations
///
struct KineticPlasmaDesc : public PlasmaDesc {
    long           Nc;                    //!< The number of simulation particles per cell.
    ShapeOrder     shape_order;           //!< The order of the shape function.
    ParticleScheme scheme;                //!< Full-f or delta-f scheme.
    Real           initial_weight;        //!< Initial particle's delta-f weight.
    Real           marker_temp_ratio;     //!< Relative fraction of marker particle temperature.
    Real           psd_refresh_frequency; //!< PSD refresh frequency.
    bool           should_refresh_psd;

    // the explicit qualifier is to prevent an accidental construction of an empty object
    //
    explicit KineticPlasmaDesc() noexcept = default;

    /// Construct a kinetic plasma description
    /// \param desc Common plasma description.
    /// \param Nc The number of simulation particles.
    /// \param shape_order Simulation particle shape order.
    /// \param psd_refresh_frequency PSD refresh frequency. Must be non-negative. Default is 0.
    /// \param scheme Whether to evolve full or delta VDF. Default is full_f.
    /// \param initial_weight Initial weight of delta-f particles. Default is 0.
    /// \param marker_temp_ratio Relative fraction of marker particle's temperature.
    ///                          Must be positive. Default is 1.
    /// \throw Any exception thrown by PlasmaDesc, and if Nc == 0.
    ///
    constexpr KineticPlasmaDesc(PlasmaDesc const &desc, unsigned Nc, ShapeOrder shape_order, Real psd_refresh_frequency,
                                ParticleScheme scheme = full_f, Real initial_weight = 0, Real marker_temp_ratio = 1)
    : PlasmaDesc(desc)
    , Nc{ Nc }
    , shape_order{ shape_order }
    , scheme{ scheme }
    , initial_weight{ full_f == scheme ? 0 : initial_weight }
    , marker_temp_ratio{ marker_temp_ratio }
    , psd_refresh_frequency{ psd_refresh_frequency }
    , should_refresh_psd{ 0.0 != psd_refresh_frequency }
    {
        if (this->Nc <= 0)
            throw std::invalid_argument{ "Nc should be positive" };
        if (this->psd_refresh_frequency < 0)
            throw std::invalid_argument{ "psd_refresh_frequency should be non-negative" };
        if (this->initial_weight < 0 || this->initial_weight > 1)
            throw std::invalid_argument{ "initial weight should be between 0 and 1 (inclusive)" };
        if (this->marker_temp_ratio <= 0)
            throw std::invalid_argument{ "relative fraction of marker particle's temperature must be a positive number" };
    }
    constexpr KineticPlasmaDesc(PlasmaDesc const &desc, unsigned Nc, ShapeOrder shape_order,
                                ParticleScheme scheme = full_f, Real initial_weight = 0, Real marker_temp_ratio = 1)
    : KineticPlasmaDesc(desc, Nc, shape_order, 0, scheme, initial_weight, marker_temp_ratio) {}

private:
    [[nodiscard]] friend constexpr auto serialize(KineticPlasmaDesc const &desc) noexcept
    {
        PlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base),
                              std::make_tuple(desc.Nc, desc.scheme, desc.initial_weight, desc.marker_temp_ratio));
    }
    [[nodiscard]] friend constexpr bool operator==(KineticPlasmaDesc const &lhs, KineticPlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Parameters for test particles
/// \details The intended use of this is debugging.
/// \tparam N The number of test particles.
template <unsigned N>
struct TestParticleDesc : public KineticPlasmaDesc {
    using Vector = MFAVector;

    static constexpr auto     number_of_test_particles = N;
    std::array<Vector, N>     vel;
    std::array<CurviCoord, N> pos;

    /// Construct a TestParticleDesc object
    /// \note It uses Nc as a placeholder for the number of test particles.
    /// \param desc Common kinetic plasma description.
    /// \param vel An array of initial velocity vectors in the field-aligned coordinates (same for the relativistic case).
    /// \param pos An array of initial curvilinear coordinates.
    ///            If the coordinate values are outside the simulation domain, those particles will be discarded.
    constexpr explicit TestParticleDesc(PlasmaDesc const &desc, std::array<Vector, N> const &vel, std::array<CurviCoord, N> const &pos)
    : KineticPlasmaDesc{ desc, N, CIC }, vel{ vel }, pos{ pos }
    {
        // reset unnecessary parameters
        op                          = 0;
        number_of_source_smoothings = 0;
    }
};

/// Parameters for a bi-Maxwellian plasma population
///
struct BiMaxPlasmaDesc : public KineticPlasmaDesc {
    Real beta1; //!< The parallel component of plasma beta.
    Real T2_T1; //!< The ratio of the perpendicular to parallel temperatures.

    /// Construct a bi-Maxwellian plasma description.
    /// \param desc Kinetic plasma description.
    /// \param beta1 Parallel plasma beta.
    /// \param T2_T1 Perpendicular-to-parallel temperature ratio. Default is 1.
    /// \throw Any exception thrown by KineticPlasmaDesc, and if either beta1 <= 0 or T2_T1 <= 0.
    ///
    explicit constexpr BiMaxPlasmaDesc(KineticPlasmaDesc const &desc, Real beta1, Real T2_T1 = 1)
    : KineticPlasmaDesc(desc), beta1{ beta1 }, T2_T1{ T2_T1 }
    {
        if (this->beta1 <= 0)
            throw std::invalid_argument{ "beta1 should be positive" };
        if (this->T2_T1 <= 0)
            throw std::invalid_argument{ "T2_T1 should be positive" };
    }

private:
    [[nodiscard]] friend constexpr auto serialize(BiMaxPlasmaDesc const &desc) noexcept
    {
        KineticPlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(desc.beta1, desc.T2_T1));
    }
    [[nodiscard]] friend constexpr bool operator==(BiMaxPlasmaDesc const &lhs, BiMaxPlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Parameters for a loss-cone distribution plasma population
/// \details The perpendicular component of the loss-cone is given by
///          f_perp = (exp(-x^2) - exp(-x^2/β)) / (1 - β)*π*θ2^2
/// where x = v2/θ2.
/// The effective perpendicular temperature is 2*T2 = (1 + β)*θ2^2.
///
struct LossconePlasmaDesc : public BiMaxPlasmaDesc {
    struct DepthWidth {
        Real beta = 0; // Loss-cone VDF β parameter.
    } losscone;

    /// Construct a loss-cone plasma description
    /// \details In this version, the effective temperatures are used to derive the necessary
    /// parameters.
    /// \param losscone Losscone β parameter. Default is 0.
    /// \param desc A bi-Maxwellian plasma description.
    /// \throw Any exception thrown by BiMaxPlasmaDesc, or if β < 0.
    ///
    explicit constexpr LossconePlasmaDesc(DepthWidth const &losscone, BiMaxPlasmaDesc const &desc)
    : BiMaxPlasmaDesc{ desc }, losscone{ losscone }
    {
        if (losscone.beta < 0)
            throw std::invalid_argument{ "losscone.beta should be non-negative" };
    }

    /// Construct a loss-cone plasma description
    /// \details In this version, the necessary parameters are explicitly specified.
    /// \param losscone Losscone β parameter. Default is 0.
    /// \param desc A kinetic plasma description.
    /// \param beta1 Parallel plasma beta.
    /// \param vth_ratio A positive number for the ratio θ2^2/θ1^2. Default is 1.
    /// \throw Any exception thrown by BiMaxPlasmaDesc, or if β < 0.
    ///
    explicit constexpr LossconePlasmaDesc(DepthWidth const &losscone, KineticPlasmaDesc const &desc, Real beta1, Real vth_ratio = 1)
    : LossconePlasmaDesc(losscone, BiMaxPlasmaDesc{ desc, beta1, (1 + losscone.beta) * vth_ratio })
    {
    }

private:
    [[nodiscard]] friend constexpr auto serialize(LossconePlasmaDesc const &desc) noexcept
    {
        BiMaxPlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(desc.losscone.beta));
    }
    [[nodiscard]] friend constexpr bool operator==(LossconePlasmaDesc const &lhs, LossconePlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Parameters for a partial shell plasma population
///
struct PartialShellPlasmaDesc : public KineticPlasmaDesc {
    Real     beta; //!< The thermal spread squared.
    unsigned zeta; //!< Exponent in pitch angle distribution, sin^ζ(α).
    Real     vs;   //!< Partial shell velocity.

    /// Construct a partial shell plasma description.
    /// \param desc Kinetic plasma description.
    /// \param beta Partial shell thermal spread squared. Must be positive.
    /// \param zeta Non-negative integer exponent in pitch angle distribution, sin^ζ(α). Default is 0.
    /// \param vs Partial shell velocity. Must be non-negative. Default is 0.
    ///           For the relativistic case, this quantity is considered to be normalized momentum.
    /// \throw Any exception thrown by KineticPlasmaDesc, and if either beta <= 0 or vs < 0.
    ///
    constexpr PartialShellPlasmaDesc(KineticPlasmaDesc const &desc, Real beta, unsigned zeta = 0, Real vs = 0)
    : KineticPlasmaDesc(desc), beta{ beta }, zeta{ zeta }, vs{ vs }
    {
        if (this->beta <= 0)
            throw std::invalid_argument{ "beta should be positive" };
        if (this->vs < 0)
            throw std::invalid_argument{ "vs should be non-negative" };
    }

private:
    [[nodiscard]] friend constexpr auto serialize(PartialShellPlasmaDesc const &desc) noexcept
    {
        KineticPlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(desc.beta, desc.zeta, desc.vs));
    }
    [[nodiscard]] friend constexpr bool operator==(PartialShellPlasmaDesc const &lhs, PartialShellPlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Parameters for a partial shell plasma population
///
struct CounterBeamPlasmaDesc : public KineticPlasmaDesc {
    Real beta; //!< The thermal spread squared.
    Real nu;   //!< Pitch angle gaussian width.
    Real vs;   //!< Partial shell velocity.

    /// Construct a partial shell plasma description.
    /// \param desc Kinetic plasma description.
    /// \param beta Partial shell thermal spread squared. Must be positive.
    /// \param nu Positive real number of the pitch angle gaussian width.
    /// \param vs Partial shell velocity. Must be non-negative. Default is 0.
    ///           For the relativistic case, this quantity is considered to be normalized momentum.
    /// \throw Any exception thrown by KineticPlasmaDesc, and if either beta <= 0, nu <= 0, or vs < 0.
    ///
    constexpr CounterBeamPlasmaDesc(KineticPlasmaDesc const &desc, Real beta, Real nu, Real vs = 0)
    : KineticPlasmaDesc(desc), beta{ beta }, nu{ nu }, vs{ vs }
    {
        if (this->beta <= 0)
            throw std::invalid_argument{ "beta should be positive" };
        if (this->nu <= 0)
            throw std::invalid_argument{ "nu should be positive" };
        if (this->vs < 0)
            throw std::invalid_argument{ "vs should be non-negative" };
    }

private:
    [[nodiscard]] friend constexpr auto serialize(CounterBeamPlasmaDesc const &desc) noexcept
    {
        KineticPlasmaDesc const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(desc.beta, desc.nu, desc.vs));
    }
    [[nodiscard]] friend constexpr bool operator==(CounterBeamPlasmaDesc const &lhs, CounterBeamPlasmaDesc const &rhs) noexcept
    {
        return serialize(lhs) == serialize(rhs);
    }
};

/// Base class of external current source descriptor
struct ExternalSourceBase : public PlasmaDesc {
    Real  omega{};  // angular frequency
    Range extent{}; // start time and duration; this excludes the ease-in/-out phases
    struct EasePhase {
        Real in{};  // ease-in duration; non-negative
        Real out{}; // ease-out duration; non-negative
        constexpr EasePhase() noexcept = default;
        constexpr EasePhase(Real in, Real out) noexcept
        : in{ in }, out{ out } {}
    } ease{};

    /// Construct an external source descriptor
    /// \details The ease-in/-out phase is to gradually ramp up and down the external source applied.
    ///          The ease-in phase starts at `start` - `ease_in` and the ease-out phase starts at `start` + `duration` + `ease_out`.
    ///          So, the total duration of the external source application is `duration` + 2*`ease_in`.
    /// \param omega The angular frequency of the external source.
    /// \param extent The start time and duration of the external source.
    ///               The ease-in/-out phases are not part of the time extent.
    /// \param ease_phase A pair of ease-in/-out durations before and after applying the source.
    ///                A non-negative value is expected.
    constexpr ExternalSourceBase(Real omega, Range extent, EasePhase ease_phase, unsigned n_smooths = {})
    : PlasmaDesc{ n_smooths }, omega{ omega }, extent{ extent }, ease{ ease_phase }
    {
        if (this->ease.in < 0)
            throw std::invalid_argument{ "ease-in should be non-negative" };
        if (this->ease.out < 0)
            throw std::invalid_argument{ "ease-out should be non-negative" };
    }

    /// Construct an external source descriptor
    /// \param omega The angular frequency of the external source.
    /// \param extent The start time and duration of the external source.
    ///               The ease-in/-out phases are not part of the time extent.
    /// \param ease_inout The ease-in/-out duration before and after applying the source.
    ///                A non-negative value is expected.
    constexpr ExternalSourceBase(Real omega, Range extent, Real ease_inout, unsigned n_smooths = {})
    : PlasmaDesc{ n_smooths }, omega{ omega }, extent{ extent }, ease{ ease_inout, ease_inout }
    {
        if (ease_inout < 0)
            throw std::invalid_argument{ "ease_inout should be non-negative" };
    }

    ExternalSourceBase() noexcept = default;

private:
    [[nodiscard]] friend constexpr auto serialize(ExternalSourceBase const &desc) noexcept
    {
        return std::make_tuple(desc.number_of_source_smoothings, desc.omega, desc.extent.loc, desc.extent.len, desc.ease.in, desc.ease.out);
    }
};

/// External current source descriptor
/// \tparam N The number of source points.
template <unsigned N>
struct ExternalSourceDesc : public ExternalSourceBase {
    static constexpr auto        number_of_source_points = N;
    std::array<ComplexVector, N> J0;  // source current amplitude (complex field-aligned components)
    std::array<CurviCoord, N>    pos; // source location

    /// Construct an external source descriptor
    /// \param base The common parameters wrapped in an ExternalSourceBase object.
    /// \param J0 An array of the complex current sources in field-aligned coordinates.
    /// \param pos An array of the curvilinear source locations.
    constexpr explicit ExternalSourceDesc(ExternalSourceBase const &base, std::array<ComplexVector, N> J0, std::array<CurviCoord, N> pos)
    : ExternalSourceBase{ base }, J0{ J0 }, pos{ pos }
    {
    }

private:
    [[nodiscard]] friend constexpr auto serialize(ExternalSourceDesc const &desc) noexcept
    {
        ExternalSourceBase const &base = desc;
        return std::tuple_cat(serialize(base), std::make_tuple(number_of_source_points));
    }
};
LIBPIC_NAMESPACE_END(1)
