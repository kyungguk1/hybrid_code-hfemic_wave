/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/VDF.h>
#include <PIC/VDF/CounterBeamVDF.h>
#include <PIC/VDF/LossconeVDF.h>
#include <PIC/VDF/MaxwellianVDF.h>
#include <PIC/VDF/PartialShellVDF.h>
#include <PIC/VDF/TestParticleVDF.h>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

LIBPIC_NAMESPACE_BEGIN(1)
class VDFVariant {
    // visitor overload facility
    //
    template <class Ret, class Lambda>
    struct Overload : Lambda {
        using Lambda::operator();

        Ret operator()(std::monostate const &) const
        {
            throw std::domain_error{ __PRETTY_FUNCTION__ };
        }
    };

    template <class Ret, class Vis>
    [[nodiscard]] static constexpr Overload<Ret, Vis> make_vis(Vis vis)
    {
        return { std::move(vis) };
    }

public:
    using variant_t = std::variant<std::monostate, MaxwellianVDF, LossconeVDF, PartialShellVDF, CounterBeamVDF, TestParticleVDF>;

    // ctor's
    //
    VDFVariant() = default;
    template <class VDF, class... Args>
    explicit VDFVariant(std::in_place_type_t<VDF> type, Args &&...args) noexcept(std::is_nothrow_constructible_v<VDF, Args...>)
    : var{ type, std::forward<Args>(args)... }
    {
    }

    template <class... Args>
    [[nodiscard]] static auto make(BiMaxPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<MaxwellianVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<MaxwellianVDF, decltype(desc), Args...>);
        return std::make_unique<VDFVariant>(std::in_place_type<MaxwellianVDF>, desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] static auto make(LossconePlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<LossconeVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<LossconeVDF, decltype(desc), Args...>);
        return std::make_unique<VDFVariant>(std::in_place_type<LossconeVDF>, desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] static auto make(PartialShellPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<PartialShellVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<PartialShellVDF, decltype(desc), Args...>);
        return std::make_unique<VDFVariant>(std::in_place_type<PartialShellVDF>, desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] static auto make(CounterBeamPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<CounterBeamVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<CounterBeamVDF, decltype(desc), Args...>);
        return std::make_unique<VDFVariant>(std::in_place_type<CounterBeamVDF>, desc, std::forward<Args>(args)...);
    }
    template <unsigned N, class... Args>
    [[nodiscard]] static auto make(TestParticleDesc<N> const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<TestParticleVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<TestParticleVDF, decltype(desc), Args...>);
        return std::make_unique<VDFVariant>(std::in_place_type<TestParticleVDF>, desc, std::forward<Args>(args)...);
    }

    template <class... Args>
    [[nodiscard]] decltype(auto) emplace(BiMaxPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<MaxwellianVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<MaxwellianVDF, decltype(desc), Args...>);
        return var.emplace<MaxwellianVDF>(desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] decltype(auto) emplace(LossconePlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<LossconeVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<LossconeVDF, decltype(desc), Args...>);
        return var.emplace<LossconeVDF>(desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] decltype(auto) emplace(PartialShellPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<PartialShellVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<PartialShellVDF, decltype(desc), Args...>);
        return var.emplace<PartialShellVDF>(desc, std::forward<Args>(args)...);
    }
    template <class... Args>
    [[nodiscard]] decltype(auto) emplace(CounterBeamPlasmaDesc const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<CounterBeamVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<CounterBeamVDF, decltype(desc), Args...>);
        return var.emplace<CounterBeamVDF>(desc, std::forward<Args>(args)...);
    }
    template <unsigned N, class... Args>
    [[nodiscard]] decltype(auto) emplace(TestParticleDesc<N> const &desc, Args &&...args) noexcept(
        std::is_nothrow_constructible_v<TestParticleVDF, decltype(desc), Args...>)
    {
        static_assert(std::is_constructible_v<TestParticleVDF, decltype(desc), Args...>);
        return var.emplace<TestParticleVDF>(desc, std::forward<Args>(args)...);
    }

    // method dispatch
    //
    [[nodiscard]] KineticPlasmaDesc const &plasma_desc() const noexcept
    {
        using Ret          = decltype(plasma_desc());
        constexpr auto vis = make_vis<Ret>([](auto const &alt) -> Ret {
            return alt.plasma_desc();
        });
        return std::visit(vis, var);
    }
    [[nodiscard]] Particle emit() const
    {
        using Ret          = decltype(emit());
        constexpr auto vis = make_vis<Ret>([](auto const &alt) -> Ret {
            return alt.emit();
        });
        return std::visit(vis, var);
    }
    [[nodiscard]] std::vector<Particle> emit(unsigned long n) const
    {
        using Ret      = decltype(emit(n));
        const auto vis = make_vis<Ret>([n](auto const &alt) -> Ret {
            return alt.emit(n);
        });
        return std::visit(vis, var);
    }

    [[nodiscard]] Scalar n0(CurviCoord const &pos) const
    {
        using Ret      = decltype(n0(pos));
        const auto vis = make_vis<Ret>([&pos](auto const &alt) -> Ret {
            return alt.n0(pos);
        });
        return std::visit(vis, var);
    }
    [[nodiscard]] CartVector nV0(CurviCoord const &pos) const
    {
        using Ret      = decltype(nV0(pos));
        const auto vis = make_vis<Ret>([&pos](auto const &alt) -> Ret {
            return alt.nV0(pos);
        });
        return std::visit(vis, var);
    }
    [[nodiscard]] CartTensor nvv0(CurviCoord const &pos) const
    {
        using Ret      = decltype(nvv0(pos));
        const auto vis = make_vis<Ret>([&pos](auto const &alt) -> Ret {
            return alt.nvv0(pos);
        });
        return std::visit(vis, var);
    }

    [[nodiscard]] Real real_f0(Particle const &ptl) const
    {
        using Ret      = decltype(real_f0(ptl));
        const auto vis = make_vis<Ret>([&ptl](auto const &alt) -> Ret {
            return alt.real_f0(ptl);
        });
        return std::visit(vis, var);
    }

    [[nodiscard]] Real Nrefcell_div_Ntotal() const
    {
        using Ret      = decltype(Nrefcell_div_Ntotal());
        const auto vis = make_vis<Ret>([](auto const &alt) -> Ret {
            return alt.Nrefcell_div_Ntotal();
        });
        return std::visit(vis, var);
    }

private:
    variant_t var;
};
LIBPIC_NAMESPACE_END(1)
