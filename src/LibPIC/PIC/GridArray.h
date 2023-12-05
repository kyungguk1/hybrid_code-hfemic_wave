/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>
#include <PIC/Shape.h>

#include <algorithm>
#include <array>
#include <iterator> // std::make_move_iterator
#include <memory>
#include <sstream>
#include <type_traits> // std::forward

LIBPIC_NAMESPACE_BEGIN(1)
/// 1D grid-point array with paddings on both ends that act as ghost cells
///
template <class Type, long Size, long Pad>
class GridArray {
    static_assert(Size > 0, "there must be at least one element");
    static_assert(Pad >= 0);

public:
    constexpr static long size() noexcept { return Size; }
    constexpr static long pad_size() noexcept { return Pad; }
    constexpr static long max_size() noexcept { return size() + 2 * pad_size(); }

    GridArray(GridArray const &)     = delete;
    GridArray(GridArray &&) noexcept = default;
    GridArray &operator=(GridArray &&) noexcept = default;

private:
    using Backend = std::array<Type, unsigned(max_size())>;
    std::unique_ptr<Backend> ptr;

public:
    GridArray()
    : ptr{ std::make_unique<Backend>() }
    {
    }

    GridArray &operator=(GridArray const &other) noexcept
    {
        if (this != &other)
            *this->ptr = *other.ptr; // other.ptr is expected to point to a valid object
        return *this;
    }

    // iterators
    //
    [[nodiscard]] Type const *dead_begin() const noexcept { return ptr->data(); }
    [[nodiscard]] Type       *dead_begin() noexcept { return ptr->data(); }
    [[nodiscard]] friend auto dead_begin(GridArray &gridArray) noexcept { return gridArray.dead_begin(); }
    [[nodiscard]] friend auto dead_begin(GridArray const &gridArray) noexcept { return gridArray.dead_begin(); }

    [[nodiscard]] Type const *dead_end() const noexcept { return ptr->data() + max_size(); }
    [[nodiscard]] Type       *dead_end() noexcept { return ptr->data() + max_size(); }
    [[nodiscard]] friend auto dead_end(GridArray &gridArray) noexcept { return gridArray.dead_end(); }
    [[nodiscard]] friend auto dead_end(GridArray const &gridArray) noexcept { return gridArray.dead_end(); }

    [[nodiscard]] Type const *begin() const noexcept { return dead_begin() + pad_size(); }
    [[nodiscard]] Type       *begin() noexcept { return dead_begin() + pad_size(); }
    [[nodiscard]] friend auto begin(GridArray &gridArray) noexcept { return gridArray.begin(); }
    [[nodiscard]] friend auto begin(GridArray const &gridArray) noexcept { return gridArray.begin(); }

    [[nodiscard]] Type const *end() const noexcept { return begin() + size(); }
    [[nodiscard]] Type       *end() noexcept { return begin() + size(); }
    [[nodiscard]] friend auto end(GridArray &gridArray) noexcept { return gridArray.end(); }
    [[nodiscard]] friend auto end(GridArray const &gridArray) noexcept { return gridArray.end(); }

    // subscripts; index relative to the first non-padding element (i.e., relative to *begin())
    //
    [[nodiscard]] Type const &operator[](long const i) const noexcept { return *(begin() + i); }
    [[nodiscard]] Type       &operator[](long const i) noexcept { return *(begin() + i); }

    /// content swap
    ///
    void swap(GridArray &o) noexcept { ptr.swap(o.ptr); }

    /// content filling (including paddings)
    ///
    void fill_all(Type const &value) noexcept { std::fill(dead_begin(), dead_end(), value); }

    /// content filling (excluding paddings)
    ///
    void fill_interior(Type const &value) noexcept { std::fill(begin(), end(), value); }

    /// apply function to interior points
    template <class Apply>
    void for_interior(Apply &&apply) & { std::for_each(begin(), end(), std::forward<Apply>(apply)); }
    template <class Apply>
    void for_interior(Apply &&apply) const & { std::for_each(begin(), end(), std::forward<Apply>(apply)); }
    template <class Apply>
    void for_interior(Apply &&apply) && { std::for_each(std::make_move_iterator(begin()), std::make_move_iterator(end()), std::forward<Apply>(apply)); }

    /// apply function to all points
    template <class Apply>
    void for_all(Apply &&apply) & { std::for_each(dead_begin(), dead_end(), std::forward<Apply>(apply)); }
    template <class Apply>
    void for_all(Apply &&apply) const & { std::for_each(dead_begin(), dead_end(), std::forward<Apply>(apply)); }
    template <class Apply>
    void for_all(Apply &&apply) && { std::for_each(std::make_move_iterator(dead_begin()), std::make_move_iterator(dead_end()), std::forward<Apply>(apply)); }

    /// grid interpolator
    ///
    template <long Order>
    [[nodiscard]] auto interp(Shape<Order> const &sx) const noexcept
    {
        static_assert(pad_size() >= Order, "padding should be greater than or equal to the shape order");
        auto y = Type{};
        for (unsigned j = 0; j <= Order; ++j) {
            y += (*this)[sx.i(j)] * sx.w(j);
        }
        return y;
    }

    /// particle deposit; in-place operation
    ///
    template <long Order, class U>
    void deposit(Shape<Order> const &sx, U const &weight) noexcept
    {
        static_assert(pad_size() >= Order, "padding should be greater than or equal to the shape order");
        for (unsigned j = 0; j <= Order; ++j) {
            (*this)[sx.i(j)] += weight * sx.w(j);
        }
    }

    /// 3-point smoothing
    ///
    decltype(auto) smooth_assign(GridArray const &source) &noexcept
    {
        GridArray &filtered = *this;
        static_assert(pad_size() >= 1, "not enough padding");
        for (long i = 0; i < size(); ++i) {
            filtered[i] = (source[i - 1] + 2 * source[i] + source[i + 1]) * .25;
        }
        return filtered;
    }
    [[nodiscard]] auto smooth_assign(GridArray const &source) &&noexcept
    {
        return std::move(this->smooth_assign(source));
    }

    // pretty print (buffered)
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, GridArray const &grid)
    {
        std::basic_ostringstream<CharT, Traits> ss;
        {
            ss.flags(os.flags());
            ss.imbue(os.getloc());
            ss.precision(os.precision());
            //
            static_assert(GridArray::size() > 0);
            ss << '{' << *grid.begin(); // guaranteed to be at least one element
            std::for_each(std::next(grid.begin()), grid.end(), [&](auto const &value) {
                ss << ", " << value;
            });
            ss << '}';
        }
        return os << ss.str();
    }
};
LIBPIC_NAMESPACE_END(1)
