/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <PIC/Config.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

LIBPIC_NAMESPACE_BEGIN(1)
/// option parser from command-line arguments
///
/// Parsed are short-style options in the form '-opt_name', which are interpreted as boolean true
/// (represented by string literal 'true'), and long-style options in the form '--opt_name=value' or
/// '--opt_name value', which are interpreted as key-value pairs. Values in the second form of the
/// long-style options must not preceed with '--'.
///
/// The short-style option '-opt_name' is equivalent to '--opt_name=true', and the string literals
/// 'true' and 'false' are interpreted as boolean, but do not cast to integers '1' and '0'. Nor can
/// integers cast to booleans.
///
/// Any number of leading/trailing, but not interspersed, whitespaces in 'opt_name' and 'value' are
/// removed before parsing. An empty string, after removing the whitespaces, as opt_name and/or
/// value is ill-formed.
///
class [[nodiscard]] Options {
public:
    enum Style : long {
        short_ = 1, //!< tag for short-style option
        long_  = 2  //!< tag for long-style option
    };

    // option value parser
    //
    class Value {
        friend Options;
        std::string m_str;
        Style       m_style{ long_ };

    public:
        Value() noexcept = default;
        Value(std::string str, Style style) noexcept
        : m_str{ std::move(str) }, m_style{ style } {}

        decltype(auto)     operator*() const noexcept { return (m_str); }
        auto              *operator->() const noexcept { return std::addressof(m_str); }
        [[nodiscard]] auto style() const noexcept { return m_style; }

        explicit operator std::string const &() const noexcept { return m_str; }
        explicit operator char const *() const noexcept { return m_str.c_str(); }
        explicit operator int() const { return std::stoi(m_str); }
        explicit operator long() const { return std::stol(m_str); }
        explicit operator unsigned long() const { return std::stoul(m_str); }
        explicit operator float() const { return std::stof(m_str); }
        explicit operator double() const { return std::stod(m_str); }
        explicit operator bool() const;

        template <class T>
        [[nodiscard]] auto as() const
        {
            return static_cast<std::decay_t<T>>(*this);
        }
        // this is to support retrieving values through std::visit
        template <class T>
        void operator()(T *p) const { *p = this->template as<T>(); }
    };

private:
    std::map<std::string, Value> opts;

public:
    [[nodiscard]] std::map<std::string, Value> const *operator->() const &noexcept { return &opts; }
    [[nodiscard]] std::map<std::string, Value> const &operator*() const &noexcept { return opts; }

    Options() noexcept = default;

    /// parses options in the argument list and returns unparsed, order-preserved, remaining
    /// arguments
    ///
    /// multiple calls will override/append to the options already parsed previously
    ///
    std::vector<std::string> parse(std::vector<std::string> args);

private:
    [[nodiscard]] static auto transform_long_style(std::vector<std::string> args) -> decltype(args);
    [[nodiscard]] static auto parse_short_options(std::vector<std::string>      args,
                                                  std::map<std::string, Value> &opts)
        -> decltype(args);
    [[nodiscard]] static auto parse_long_options(std::vector<std::string>      args,
                                                 std::map<std::string, Value> &opts)
        -> decltype(args);

    // pretty print
    //
    template <class CharT, class Traits>
    friend decltype(auto) operator<<(std::basic_ostream<CharT, Traits> &os, Options const &opts)
    {
        auto const printer = [](decltype(os) os, auto const &kv) -> decltype(auto) {
            auto const &[key, val] = kv;
            return os << key << " : " << *val;
        };
        os << '{';
        if (!opts->empty()) {
            printer(os, *begin(*opts));
            std::for_each(std::next(begin(*opts)), end(*opts), [&os, printer](auto const &kv) {
                printer(os << ", ", kv);
            });
        }
        return os << '}';
    }
};
LIBPIC_NAMESPACE_END(1)
