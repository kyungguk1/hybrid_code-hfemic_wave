/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Options.h"

#include <stdexcept>

LIBPIC_NAMESPACE_BEGIN(1)
Options::Value::operator bool() const
{
    if (m_str == "true")
        return true;
    else if (m_str == "false")
        return false;
    throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ }
                                 + " - invalid string literal for boolean : " + m_str };
}

namespace {
[[nodiscard]] std::string trim(std::string s) noexcept(noexcept(s.erase(end(s), end(s))))
{
    auto const pred = [](auto const c) noexcept {
        return ' ' != c;
    };
    s.erase(begin(s), std::find_if(begin(s), end(s), pred));        // leading whitespace(s)
    s.erase(std::find_if(rbegin(s), rend(s), pred).base(), end(s)); // trailing whitespace(s)
    return s;
}
} // namespace
std::vector<std::string> Options::parse(std::vector<std::string> args)
{
    args = transform_long_style(std::move(args));
    args = parse_short_options(std::move(args), opts);
    args = parse_long_options(std::move(args), opts);
    return args;
}
auto Options::transform_long_style(std::vector<std::string> args) -> decltype(args)
{
    if (size(args) > 1) { // at least two entries
        for (auto key = rbegin(args), val = key++; key != rend(args); val = key++) {
            if (bool const condition = key->size() > 2 && ((*key)[0] == '-' && (*key)[1] == '-')
                                    && key->find('=') == key->npos;
                !condition) {
                continue;
            }
            if (val->size() >= 2 && ((*val)[0] == '-' && (*val)[1] == '-')) {
                throw std::invalid_argument{ std::string{ __PRETTY_FUNCTION__ }
                                             + " - long-style option `" + *key + ' ' + *val
                                             + "' is ill-formed" };
            }

            // append value part to the key and erase it
            (*key += '=') += *val;
            args.erase(key.base());
        }
    }
    return args;
}
auto Options::parse_short_options(std::vector<std::string> args, std::map<std::string, Value> &opts)
    -> decltype(args)
{
    // parse short options whose form is -opt_name which is equivalent to --opt_name=1
    //
    auto const  first  = std::stable_partition(begin(args), end(args), [](std::string const &s) {
        return !(s.size() > 1 && (s[0] == '-' && s[1] != '-'));
      });
    char const *prefix = __PRETTY_FUNCTION__;
    auto        parser = [&opts, prefix](std::string const &s) -> void {
        if (auto name = trim(s.substr(1)); !name.empty()) {
            opts[std::move(name)] = { "true", short_ };
            return;
        }
        throw std::invalid_argument{ std::string{ prefix } + " - short-style option `" + s
                                     + "' is ill-formed" };
    };
    std::for_each(first, end(args), parser);
    args.erase(first, end(args));
    //
    return args;
}
auto Options::parse_long_options(std::vector<std::string> args, std::map<std::string, Value> &opts)
    -> decltype(args)
{
    // parse long options whose form is --opt_name=value
    //
    auto const  first  = std::stable_partition(begin(args), end(args), [](std::string const &s) {
        return !(s.size() > 2 && (s[0] == '-' && s[1] == '-'));
      });
    char const *prefix = __PRETTY_FUNCTION__;
    auto        parser = [&opts, prefix](std::string s) -> void {
        s = s.substr(2);
        if (auto const pos = s.find('='); pos != s.npos) {
            if (auto name = trim(s.substr(0, pos)); !name.empty()) {
                if (auto value = trim(s.substr(pos + 1)); !value.empty()) {
                    opts[std::move(name)] = { std::move(value), long_ };
                    return;
                }
            }
        }
        throw std::invalid_argument{ std::string{ prefix } + " - long-style option `--" + s
                                     + "' is ill-formed" };
    };
    std::for_each(first, end(args), parser);
    args.erase(first, end(args));
    //
    return args;
}
LIBPIC_NAMESPACE_END(1)
