/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/BorisPush.h>
#include <PIC/CartCoord.h>
#include <PIC/UTL/println.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

namespace {
template <class T>
[[maybe_unused]] void printer(std::vector<T> const &vs)
{
    if (vs.empty()) {
        std::puts("{}");
    } else {
        print(std::cout, "{\n    ", vs.front());
        std::for_each(std::next(begin(vs)), end(vs), [](T const &v) {
            print(std::cout, ",\n    ", v);
        });
        std::puts("\n}");
    }
}
} // namespace

TEST_CASE("Test LibPIC::NonrelativisticBorisPush", "[LibPIC::NonrelativisticBorisPush]")
{
    unsigned const  nt = 360;
    Real const      dt = 1;
    Real const      c  = 1;
    Real const      O0 = 1;
    Real const      Oc = 2 * M_PI / nt;
    BorisPush const boris{ dt, c, O0, Oc };

    std::vector<CartVector> vs;
    std::generate_n(std::back_inserter(vs), nt,
                    [&boris, v = CartVector{ 1, 0, 0 }]() mutable {
                        constexpr CartVector B0{ 0, 0, 1 };
                        boris.non_relativistic(v, B0, {});
                        return v;
                    });
    // printer(vs);
    CHECK(std::accumulate(begin(vs), end(vs), true, [](bool lhs, CartVector const &v) {
        return lhs && std::abs(dot(v, v) - 1) < 1e-15 && v.z == 0;
    }));

    vs = { { 0, 0, 0 } };
    for (long t = 0; t < nt; ++t) {
        CartVector const E{ 0, 0, std::cos(2 * M_PI * (Real(t) + dt / 2) / nt) };
        boris.non_relativistic(vs.emplace_back(vs.back()), {}, E / c);
    }
    vs.erase(begin(vs));
    // printer(vs);
    CHECK(std::accumulate(begin(vs), end(vs), true, [](bool lhs, CartVector const &v) {
        return lhs && v.x == 0 && v.y == 0;
    }));
    auto const rms = std::accumulate(begin(vs), end(vs), Real{}, [](Real const lhs, CartVector const &v) {
                         return lhs + dot(v, v);
                     })
                   / Real(vs.size());
    CHECK(rms == Approx{ 0.50001269258580949284 }.epsilon(1e-10));
}

TEST_CASE("Test LibPIC::NonrelativisticBorisPush::SimpleHarmonicOscillator", "[LibPIC::NonrelativisticBorisPush::SimpleHarmonicOscillator]")
{
    unsigned const  nt = 360;
    Real const      w0 = 2 * M_PI;
    Real const      dt = 2 * M_PI / w0 / nt;
    Real const      c  = 1;
    Real const      O0 = 1;
    Real const      Oc = 1;
    BorisPush const boris{ dt, c, O0, Oc };

    auto const x0 = CartCoord{ 0 };
    auto const v0 = CartVector{ 1, 0, 0 };

    std::vector<CartCoord>  xs{ x0 };
    std::vector<CartVector> vs{ v0 };
    xs.back() += 0.5 * dt * vs.back().x;
    for (unsigned i = 0; i < nt; ++i) {
        auto const F = -w0 * w0 * CartVector{ xs.back().x, 0, 0 };
        boris.non_relativistic(vs.emplace_back(vs.back()), {}, F / c);
        xs.emplace_back(xs.back()) += dt * vs.back().x;
    }

    auto const xs2 = std::accumulate(std::next(begin(xs)), end(xs), Real{}, [](Real const &sum, CartCoord const &x) {
                         return sum + (x * x).x;
                     })
                   * w0 * w0 / nt;
    CHECK(xs2 == Approx{ 0.5 }.epsilon(1e-4));
    auto const vs2 = std::accumulate(std::next(begin(vs)), end(vs), Real{}, [](Real const &sum, CartVector const &v) {
                         return sum + dot(v, v);
                     })
                   / nt;
    CHECK(vs2 == Approx{ 0.5 }.epsilon(1e-4));
}

TEST_CASE("Test LibPIC::RelativisticBorisPush::NonRelativistic", "[LibPIC::RelativisticBorisPush::NonRelativistic]")
{
    unsigned const  nt = 360;
    Real const      dt = 1;
    Real const      c  = 1e3;
    Real const      O0 = 1;
    Real const      Oc = 2 * M_PI / nt;
    BorisPush const boris{ dt, c, O0, Oc };

    std::vector<FourCartVector> gcgvs;
    std::generate_n(std::back_inserter(gcgvs), nt, [&boris, gcgv = lorentz_boost<-1>(FourCartVector{ c, {} }, 1 / c)]() mutable {
        constexpr CartVector B0{ 0, 0, 1 };
        boris.relativistic(gcgv, B0, {});
        return gcgv;
    });
    // printer(gcgvs);
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true,
                          [gamma = std::sqrt(1 + 1 / (c * c))](bool lhs, FourCartVector const &gcgv) {
                              auto const beta = std::sqrt(dot(gcgv.s, gcgv.s)) / Real{ gcgv.t };
                              return lhs
                                  && std::abs(1 / std::sqrt((1 - beta) * (1 + beta)) - gamma) < 1e-15
                                  && gcgv.s.z == 0;
                          }));
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true,
                          [c, gamma = std::sqrt(1 + 1 / (c * c))](bool lhs, FourCartVector const &gcgv) {
                              auto const g = *gcgv.t / c;
                              return lhs && std::abs(g - gamma) / gamma < 1e-15;
                          }));

    gcgvs.clear();
    gcgvs.emplace_back(c, CartVector{});
    for (long t = 0; t < nt; ++t) {
        CartVector const E{ 0, 0, std::cos(2 * M_PI * (Real(t) + dt / 2) / nt) };
        boris.relativistic(gcgvs.emplace_back(gcgvs.back()), {}, E / c);
    }
    gcgvs.erase(begin(gcgvs));
    // printer(gcgvs);
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true, [](bool lhs, FourCartVector const &gcgv) {
        return lhs && gcgv.s.x == 0 && gcgv.s.y == 0;
    }));
    auto const rms = std::accumulate(begin(gcgvs), end(gcgvs), Real{}, [](Real const lhs, FourCartVector const &gcgv) {
                         return lhs + dot(gcgv.s, gcgv.s);
                     })
                   / Real(gcgvs.size());
    CHECK(rms == Approx{ 0.50001269258580949284 }.epsilon(1e-10));
}

TEST_CASE("Test LibPIC::RelativisticBorisPush::RelativisticRegime", "[LibPIC::RelativisticBorisPush::RelativisticRegime]")
{
    unsigned const  nt = 360;
    Real const      dt = 1;
    Real const      c  = 4;
    Real const      O0 = 1;
    Real const      Oc = 2 * M_PI / nt;
    BorisPush const boris{ dt, c, O0, Oc };

    std::vector<FourCartVector> gcgvs;
    std::generate_n(std::back_inserter(gcgvs), nt, [&boris, gcgv = lorentz_boost<-1>(FourCartVector{ c, {} }, 1 / c)]() mutable {
        constexpr CartVector B0{ 0, 0, 1 };
        boris.relativistic(gcgv, B0, {});
        return gcgv;
    });
    // printer(gcgvs);
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true,
                          [gamma = std::sqrt(1 + 1 / (c * c))](bool lhs, FourCartVector const &gcgv) {
                              auto const beta = std::sqrt(dot(gcgv.s, gcgv.s)) / Real{ gcgv.t };
                              return lhs
                                  && std::abs(1 / std::sqrt((1 - beta) * (1 + beta)) - gamma) < 1e-15
                                  && gcgv.s.z == 0;
                          }));
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true,
                          [c, gamma = std::sqrt(1 + 1 / (c * c))](bool lhs, FourCartVector const &gcgv) {
                              auto const g = *gcgv.t / c;
                              return lhs && std::abs(g - gamma) / gamma < 1e-15;
                          }));

    gcgvs.clear();
    gcgvs.emplace_back(c, CartVector{});
    for (long t = 0; t < nt; ++t) {
        CartVector const E{ 0, 0, std::cos(2 * M_PI * (Real(t) + dt / 2) / nt) };
        boris.relativistic(gcgvs.emplace_back(gcgvs.back()), {}, E / c);
    }
    gcgvs.erase(begin(gcgvs));
    // printer(gcgvs);
    CHECK(std::accumulate(begin(gcgvs), end(gcgvs), true, [](bool lhs, FourCartVector const &gcgv) {
        return lhs && gcgv.s.x == 0 && gcgv.s.y == 0;
    }));
    auto const rms = std::accumulate(begin(gcgvs), end(gcgvs), Real{}, [](Real const lhs, FourCartVector const &gcgv) {
                         return lhs + dot(gcgv.s, gcgv.s);
                     })
                   / Real(gcgvs.size());
    CHECK(rms == Approx{ 0.50001269258580949284 }.epsilon(1e-10));
}

TEST_CASE("Test LibPIC::RelativisticBorisPush::SimpleHarmonicOscillator", "[LibPIC::RelativisticBorisPush::SimpleHarmonicOscillator]")
{
    unsigned const  nt = 360;
    Real const      w0 = 2 * M_PI;
    Real const      dt = 2 * M_PI / w0 / nt;
    Real const      c  = 0.5;
    Real const      O0 = 1;
    Real const      Oc = 1;
    BorisPush const boris{ dt, c, O0, Oc };

    auto const x0 = CartCoord{ 0 };
    auto const u0 = CartVector{ 1, 0, 0 };

    std::vector<CartCoord>      xs{ x0 };
    std::vector<FourCartVector> gcgvs{ lorentz_boost<-1>(FourCartVector{ c, {} }, u0 / c) };
    xs.back() += 0.5 * dt * (gcgvs.back().s * c / *gcgvs.back().t).x;
    for (unsigned i = 0; i < nt; ++i) {
        auto const F = -w0 * w0 * CartVector{ xs.back().x, 0, 0 };
        boris.relativistic(gcgvs.emplace_back(gcgvs.back()), {}, F / c);
        xs.emplace_back(xs.back()) += dt * (gcgvs.back().s * c / *gcgvs.back().t).x;
    }

    // print(xs);
    // print(gcgvs);

    auto const xs2 = std::accumulate(begin(xs), std::prev(end(xs)), Real{}, [](Real const &sum, CartCoord const &x) {
                         return sum + (x * x).x;
                     })
                   * w0 * w0 / nt;
    CHECK(xs2 == Approx{ 0.25141498816159124630 }.epsilon(1e-5));
    auto const vs2 = std::accumulate(std::next(begin(gcgvs)), end(gcgvs), Real{}, [](Real const &sum, FourCartVector const &gcgv) {
                         return sum + dot(gcgv.s, gcgv.s);
                     })
                   / nt;
    CHECK(vs2 == Approx{ 0.54204287588343380566 }.epsilon(1e-5));
}
