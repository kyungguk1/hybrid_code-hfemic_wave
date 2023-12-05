/*
 * Copyright (c) 2020-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "lippincott.h"

#include <ParallelKit/ParallelKit.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <execinfo.h>

LIBPIC_NAMESPACE_BEGIN(1)
void print_backtrace()
{
    constexpr unsigned             stack_size = 64;
    std::array<void *, stack_size> array{};
    int const                      size = backtrace(array.data(), std::size(array));
    if (char **const strings = backtrace_symbols(array.data(), size)) {
        std::for_each_n(strings, size, &std::puts);
        free(strings);
    }
}

void fatal_error(char const *reason) noexcept
{
    std::puts(reason);
    print_backtrace();
    if (parallel::mpi::Comm::is_initialized())
        MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
void fatal_error(std::string const &reason) noexcept
{
    fatal_error(reason.c_str());
}

void lippincott() noexcept
try {
    throw;
} catch (std::exception const &e) {
    fatal_error(e.what());
} catch (...) {
    fatal_error("Unknown exception");
}
LIBPIC_NAMESPACE_END(1)
