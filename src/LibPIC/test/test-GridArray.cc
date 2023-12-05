/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/GridArray.h>
#include <algorithm>
#include <numeric>

constexpr long Size = 10;

TEST_CASE("Test LibPIC::GridArray::Size0", "[LibPIC::GridArray::Size0]")
{
    constexpr long Pad = 0;
    using Grid         = GridArray<Real, Size, Pad>;
    REQUIRE(Grid::size() == Size);
    REQUIRE(Grid::pad_size() == Pad);
    REQUIRE(Grid::max_size() == Grid::size() + Grid::pad_size() * 2);
    {
        // iterators and element access
        Grid        g;
        Grid const &cg = g;
        REQUIRE(g.dead_begin() != nullptr);
        REQUIRE(std::distance(g.dead_begin(), g.dead_end()) == g.max_size());
        REQUIRE(std::distance(g.dead_begin(), g.begin()) == g.pad_size());
        REQUIRE(std::distance(g.end(), g.dead_end()) == g.pad_size());
        REQUIRE(std::distance(g.begin(), g.end()) == g.size());
        REQUIRE((g.dead_begin() == cg.dead_begin() && g.dead_end() == cg.dead_end()));
        REQUIRE((g.begin() == cg.begin() && g.end() == cg.end()));
        for (long i = -g.pad_size(); i < g.size() + g.pad_size(); ++i) {
            REQUIRE(&g[i] == std::next(g.begin(), i));
            REQUIRE(&cg[i] == &g[i]);
        }
        REQUIRE(dead_begin(g) == g.dead_begin());
        REQUIRE(dead_begin(cg) == cg.dead_begin());
        REQUIRE(dead_end(g) == g.dead_end());
        REQUIRE(dead_end(cg) == cg.dead_end());
        REQUIRE(begin(g) == g.begin());
        REQUIRE(begin(cg) == cg.begin());
        REQUIRE(end(g) == g.end());
        REQUIRE(end(cg) == cg.end());

        // fill
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        g.fill_interior(10);
        CHECK(std::accumulate(g.begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        g.fill_all(10);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        // copy/move/swap
        std::generate(g.dead_begin(), g.dead_end(), [i = 10]() mutable {
            return i++;
        });
        Grid g2;
        g2 = cg;
        REQUIRE(std::equal(cg.dead_begin(), cg.dead_end(), g2.dead_begin(), g2.dead_end()));
        g2.fill_all(10);
        g = std::move(g2);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g3{ std::move(g) };
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g4;
        std::swap(g3, g4);
        CHECK(std::accumulate(g4.dead_begin(), g4.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
    }
    {
        Grid grid;
        grid.fill_all(100);
        Grid const &cgrid       = grid;
        auto const  integer_sum = [](long const n) noexcept {
            return n * (n + 1) / 2.0;
        };
        Real sum;

        grid.for_interior([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        grid = Grid{};
        grid.fill_all(100);
        grid.for_all([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));
    }
}

TEST_CASE("Test LibPIC::GridArray::Size1", "[LibPIC::GridArray::Size1]")
{
    constexpr long Pad = 1;
    using Grid         = GridArray<Real, Size, Pad>;
    REQUIRE(Grid::size() == Size);
    REQUIRE(Grid::pad_size() == Pad);
    REQUIRE(Grid::max_size() == Grid::size() + Grid::pad_size() * 2);
    {
        // iterators and element access
        Grid        g;
        Grid const &cg = g;
        REQUIRE(g.dead_begin() != nullptr);
        REQUIRE(std::distance(g.dead_begin(), g.dead_end()) == g.max_size());
        REQUIRE(std::distance(g.dead_begin(), g.begin()) == g.pad_size());
        REQUIRE(std::distance(g.end(), g.dead_end()) == g.pad_size());
        REQUIRE(std::distance(g.begin(), g.end()) == g.size());
        REQUIRE((g.dead_begin() == cg.dead_begin() && g.dead_end() == cg.dead_end()));
        REQUIRE((g.begin() == cg.begin() && g.end() == cg.end()));
        for (long i = -g.pad_size(); i < g.size() + g.pad_size(); ++i) {
            REQUIRE(&g[i] == std::next(g.begin(), i));
            REQUIRE(&cg[i] == &g[i]);
        }
        REQUIRE(dead_begin(g) == g.dead_begin());
        REQUIRE(dead_begin(cg) == cg.dead_begin());
        REQUIRE(dead_end(g) == g.dead_end());
        REQUIRE(dead_end(cg) == cg.dead_end());
        REQUIRE(begin(g) == g.begin());
        REQUIRE(begin(cg) == cg.begin());
        REQUIRE(end(g) == g.end());
        REQUIRE(end(cg) == cg.end());

        // fill
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        g.fill_interior(10);
        CHECK(std::accumulate(g.begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        g.fill_all(10);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        // copy/move/swap
        std::generate(g.dead_begin(), g.dead_end(), [i = 10]() mutable {
            return i++;
        });
        Grid g2;
        g2 = cg;
        REQUIRE(std::equal(cg.dead_begin(), cg.dead_end(), g2.dead_begin(), g2.dead_end()));
        g2.fill_all(10);
        g = std::move(g2);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g3{ std::move(g) };
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g4;
        std::swap(g3, g4);
        CHECK(std::accumulate(g4.dead_begin(), g4.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
    }
    {
        Grid grid;
        grid.fill_all(100);
        Grid const &cgrid       = grid;
        auto const  integer_sum = [](long const n) noexcept {
            return n * (n + 1) / 2.0;
        };
        Real sum;

        grid.for_interior([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        grid = Grid{};
        grid.fill_all(100);
        grid.for_all([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));
    }
    { // smooth
        Grid source;
        source[4] = 1;

        Grid filtered;
        CHECK(&filtered == &filtered.smooth_assign(source));
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));

        filtered = Grid{}.smooth_assign(source);
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));
    }
    { // Shape<1>
        using Shape = PIC::Shape<1>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-1] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() - 1e-15 }; // to avoid out-of-range memory access
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == Approx{ 0 }.margin(1e-10);
        }));
        CHECK(*g.end() == Approx{ weight }.epsilon(1e-10));

        g.fill_all(0);
        sh = Shape{ 4.14 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
    }
}

TEST_CASE("Test LibPIC::GridArray::Size2", "[LibPIC::GridArray::Size2]")
{
    constexpr long Pad = 2;
    using Grid         = GridArray<Real, Size, Pad>;
    REQUIRE(Grid::size() == Size);
    REQUIRE(Grid::pad_size() == Pad);
    REQUIRE(Grid::max_size() == Grid::size() + Grid::pad_size() * 2);
    {
        // iterators and element access
        Grid        g;
        Grid const &cg = g;
        REQUIRE(g.dead_begin() != nullptr);
        REQUIRE(std::distance(g.dead_begin(), g.dead_end()) == g.max_size());
        REQUIRE(std::distance(g.dead_begin(), g.begin()) == g.pad_size());
        REQUIRE(std::distance(g.end(), g.dead_end()) == g.pad_size());
        REQUIRE(std::distance(g.begin(), g.end()) == g.size());
        REQUIRE((g.dead_begin() == cg.dead_begin() && g.dead_end() == cg.dead_end()));
        REQUIRE((g.begin() == cg.begin() && g.end() == cg.end()));
        for (long i = -g.pad_size(); i < g.size() + g.pad_size(); ++i) {
            REQUIRE(&g[i] == std::next(g.begin(), i));
            REQUIRE(&cg[i] == &g[i]);
        }
        REQUIRE(dead_begin(g) == g.dead_begin());
        REQUIRE(dead_begin(cg) == cg.dead_begin());
        REQUIRE(dead_end(g) == g.dead_end());
        REQUIRE(dead_end(cg) == cg.dead_end());
        REQUIRE(begin(g) == g.begin());
        REQUIRE(begin(cg) == cg.begin());
        REQUIRE(end(g) == g.end());
        REQUIRE(end(cg) == cg.end());

        // fill
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        g.fill_interior(10);
        CHECK(std::accumulate(g.begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        g.fill_all(10);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        // copy/move/swap
        std::generate(g.dead_begin(), g.dead_end(), [i = 10]() mutable {
            return i++;
        });
        Grid g2;
        g2 = cg;
        REQUIRE(std::equal(cg.dead_begin(), cg.dead_end(), g2.dead_begin(), g2.dead_end()));
        g2.fill_all(10);
        g = std::move(g2);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g3{ std::move(g) };
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g4;
        std::swap(g3, g4);
        CHECK(std::accumulate(g4.dead_begin(), g4.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
    }
    {
        Grid grid;
        grid.fill_all(100);
        Grid const &cgrid       = grid;
        auto const  integer_sum = [](long const n) noexcept {
            return n * (n + 1) / 2.0;
        };
        Real sum;

        grid.for_interior([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        grid = Grid{};
        grid.fill_all(100);
        grid.for_all([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));
    }
    { // smooth
        Grid source;
        source[4] = 1;

        Grid filtered;
        CHECK(&filtered == &filtered.smooth_assign(source));
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));

        filtered = Grid{}.smooth_assign(source);
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));
    }
    { // Shape<1>
        using Shape = PIC::Shape<1>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-1] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(*g.end() == weight);

        g.fill_all(0);
        sh = Shape{ 4.14 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight
              < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
    }
    { // Shape<2>
        using Shape = PIC::Shape<2>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(std::next(g.begin()), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-2] + g[-1] + g[0] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), std::prev(g.end()), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(std::accumulate(std::prev(g.end()), g.dead_end(), Real{}) == weight);

        g.fill_all(0);
        sh = Shape{ 4.14 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(weight * sh.w<2>() == g[sh.i<2>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight
              < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
    }
}

TEST_CASE("Test LibPIC::GridArray::Size3", "[LibPIC::GridArray::Size3]")
{
    constexpr long Pad = 3;
    using Grid         = GridArray<Real, Size, Pad>;
    REQUIRE(Grid::size() == Size);
    REQUIRE(Grid::pad_size() == Pad);
    REQUIRE(Grid::max_size() == Grid::size() + Grid::pad_size() * 2);
    {
        // iterators and element access
        Grid        g;
        Grid const &cg = g;
        REQUIRE(g.dead_begin() != nullptr);
        REQUIRE(std::distance(g.dead_begin(), g.dead_end()) == g.max_size());
        REQUIRE(std::distance(g.dead_begin(), g.begin()) == g.pad_size());
        REQUIRE(std::distance(g.end(), g.dead_end()) == g.pad_size());
        REQUIRE(std::distance(g.begin(), g.end()) == g.size());
        REQUIRE((g.dead_begin() == cg.dead_begin() && g.dead_end() == cg.dead_end()));
        REQUIRE((g.begin() == cg.begin() && g.end() == cg.end()));
        for (long i = -g.pad_size(); i < g.size() + g.pad_size(); ++i) {
            REQUIRE(&g[i] == std::next(g.begin(), i));
            REQUIRE(&cg[i] == &g[i]);
        }
        REQUIRE(dead_begin(g) == g.dead_begin());
        REQUIRE(dead_begin(cg) == cg.dead_begin());
        REQUIRE(dead_end(g) == g.dead_end());
        REQUIRE(dead_end(cg) == cg.dead_end());
        REQUIRE(begin(g) == g.begin());
        REQUIRE(begin(cg) == cg.begin());
        REQUIRE(end(g) == g.end());
        REQUIRE(end(cg) == cg.end());

        // fill
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        g.fill_interior(10);
        CHECK(std::accumulate(g.begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        g.fill_all(10);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        // copy/move/swap
        std::generate(g.dead_begin(), g.dead_end(), [i = 10]() mutable {
            return i++;
        });
        Grid g2;
        g2 = cg;
        REQUIRE(std::equal(cg.dead_begin(), cg.dead_end(), g2.dead_begin(), g2.dead_end()));
        g2.fill_all(10);
        g = std::move(g2);
        CHECK(std::accumulate(g.dead_begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g3{ std::move(g) };
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));

        Grid g4;
        std::swap(g3, g4);
        CHECK(std::accumulate(g4.dead_begin(), g4.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 10;
        }));
        CHECK(std::accumulate(g3.dead_begin(), g3.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
    }
    {
        Grid grid;
        grid.fill_all(100);
        Grid const &cgrid       = grid;
        auto const  integer_sum = [](long const n) noexcept {
            return n * (n + 1) / 2.0;
        };
        Real sum;

        grid.for_interior([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_interior([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.size()) == Approx{ sum }.epsilon(1e-15));

        grid = Grid{};
        grid.fill_all(100);
        grid.for_all([i = 1](auto &x) mutable {
            x = i++;
        });

        sum = 0;
        cgrid.for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));

        sum = 0;
        std::move(grid).for_all([&sum](auto const &x) {
            sum += x;
        });
        CHECK(integer_sum(grid.max_size()) == Approx{ sum }.epsilon(1e-15));
    }
    { // smooth
        Grid source;
        source[4] = 1;

        Grid filtered;
        CHECK(&filtered == &filtered.smooth_assign(source));
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));

        filtered = Grid{}.smooth_assign(source);
        CHECK(std::accumulate(filtered.dead_begin(), filtered.dead_end(), Real{}) == Approx{ 1 }.epsilon(1e-15));
        std::accumulate(filtered.dead_begin(), std::next(filtered.begin(), 4), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        std::accumulate(std::next(filtered.begin(), 7), filtered.dead_end(), true,
                        [](bool lhs, auto rhs) {
                            return lhs && rhs == 0;
                        });
        CHECK(.25 == Approx{ filtered[3] }.epsilon(1e-15));
        CHECK(.5 == Approx{ filtered[4] }.epsilon(1e-15));
        CHECK(.25 == Approx{ filtered[5] }.epsilon(1e-15));
    }
    { // Shape<1>
        using Shape = PIC::Shape<1>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.begin(), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-1] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), g.end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(*g.end() == weight);

        g.fill_all(0);
        sh = Shape{ 4.14 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight
              < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
    }
    { // Shape<2>
        using Shape = PIC::Shape<2>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(std::next(g.begin()), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-2] + g[-1] + g[0] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), std::prev(g.end()), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(std::accumulate(std::prev(g.end()), g.dead_end(), Real{}) == weight);

        g.fill_all(0);
        sh = Shape{ 4.14 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(weight * sh.w<2>() == g[sh.i<2>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight
              < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<0>() == g.interp(sh));
    }
    { // Shape<3>
        using Shape = PIC::Shape<3>;
        Shape      sh;
        Real const weight = 10;
        Grid       g;

        g.fill_all(0);
        sh = Shape{ -1 };
        g.deposit(sh, weight);
        CHECK(std::accumulate(std::next(g.begin()), g.dead_end(), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(g[-3] + g[-2] + g[-1] + g[0] == weight);

        g.fill_all(0);
        sh = Shape{ g.size() };
        g.deposit(sh, weight);
        CHECK(std::accumulate(g.dead_begin(), std::prev(g.end(), 2), true, [](bool lhs, auto rhs) {
            return lhs && rhs == 0;
        }));
        CHECK(std::accumulate(std::prev(g.end(), 2), g.dead_end(), Real{}) == weight);

        g.fill_all(0);
        sh = Shape{ 4 + .1 };
        g.deposit(sh, weight);
        CHECK(weight * sh.w<0>() == g[sh.i<0>()]);
        CHECK(weight * sh.w<1>() == g[sh.i<1>()]);
        CHECK(weight * sh.w<2>() == g[sh.i<2>()]);
        CHECK(weight * sh.w<3>() == g[sh.i<3>()]);
        CHECK(std::abs(weight - std::accumulate(g.dead_begin(), g.dead_end(), Real{})) / weight
              < 1e-15);

        g.fill_all(0);
        g[4] = weight;
        sh   = Shape{ 3.9 };
        CHECK(weight * sh.w<2>() == g.interp(sh));
        sh = Shape{ 4.1 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
        sh = Shape{ 4.9 };
        CHECK(weight * sh.w<1>() == g.interp(sh));
    }
}
