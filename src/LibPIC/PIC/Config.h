/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

// specify version of dependent library
//
#define PARALLELKIT_INLINE_VERSION 1
#define HDF5KIT_INLINE_VERSION     1

// unique variable name
//
#ifndef LIBPIC_UNIQUE_NAME
#define _LIBPIC_UNIQUE_NAME_(x, y) x##y
#define _LIBPIC_UNIQUE_NAME(x, y)  _LIBPIC_UNIQUE_NAME_(x, y)
#define LIBPIC_UNIQUE_NAME(base)   _LIBPIC_UNIQUE_NAME(base, __LINE__)
#endif

/// @cond
// macro operation helpers
// https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
//
#define _LIBPIC_CAT(first, ...)           _LIBPIC_PRIMITIVE_CAT(first, __VA_ARGS__)
#define _LIBPIC_PRIMITIVE_CAT(first, ...) first##__VA_ARGS__

#define _LIBPIC_CHECK_N(x, n, ...) n
#define _LIBPIC_CHECK(...)         _LIBPIC_CHECK_N(__VA_ARGS__, 0, )
#define _LIBPIC_PROBE(x)           x, 1,

#define _LIBPIC_PAIR_1_1 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_2_2 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_3_3 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_4_4 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_5_5 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_6_6 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_7_7 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_8_8 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR_9_9 _LIBPIC_PROBE(~)
#define _LIBPIC_PAIR(x, y) \
    _LIBPIC_CAT(_LIBPIC_CAT(_LIBPIC_CAT(_LIBPIC_PAIR_, x), _), y)
#define _LIBPIC_IS_SAME(x, y) _LIBPIC_CHECK(_LIBPIC_PAIR(x, y))
/// @endcond

// version namespace
//
#define LIBPIC_VERSION_NAMESPACE(ver) _LIBPIC_CAT(_LIBPIC_CAT(v_, ver), _)
/// @cond
#define _LIBPIC_NAMESPACE_VERSION_0(ver) namespace LIBPIC_VERSION_NAMESPACE(ver)
#define _LIBPIC_NAMESPACE_VERSION_1(ver) inline namespace LIBPIC_VERSION_NAMESPACE(ver)
#define _LIBPIC_NAMESPACE_VERSION(ver)                       \
    _LIBPIC_CAT(_LIBPIC_NAMESPACE_VERSION_,                  \
                _LIBPIC_IS_SAME(ver, LIBPIC_INLINE_VERSION)) \
    (ver)
#define LIBPIC_VERSION_NAMESPACE_BEGIN(ver) \
    _LIBPIC_NAMESPACE_VERSION(ver)          \
    {
#define LIBPIC_VERSION_NAMESPACE_END(ver) \
    }
/// @endcond

// root namespace
//
#define LIBPIC_ROOT_NAMESPACE PIC
#define LIBPIC_NAMESPACE_BEGIN(ver)          \
    inline namespace LIBPIC_ROOT_NAMESPACE { \
    LIBPIC_VERSION_NAMESPACE_BEGIN(ver)
#define LIBPIC_NAMESPACE_END(ver) \
    }                             \
    }
#define LIBPIC_NAMESPACE(ver) \
    LIBPIC_ROOT_NAMESPACE::LIBPIC_VERSION_NAMESPACE(ver)
