/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Driver.h"
#include <PIC/UTL/lippincott.h>
#include <PIC/UTL/println.h>

#include <future>
#include <iostream>
#include <stdexcept>

int main(int argc, char *argv[])
try {
    using parallel::mpi::Comm;
    {
        constexpr bool enable_mpi_thread = false;
        int const      required          = enable_mpi_thread ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
        int            provided;
        if (Comm::init(&argc, &argv, required, provided) != MPI_SUCCESS) {
            println(std::cout, "%% ", __PRETTY_FUNCTION__, " - mpi::Comm::init(...) returned error");
            return 1;
        }
        if (provided < required) {
            println(std::cout, "%% ", __PRETTY_FUNCTION__, " - provided < required");
            return 1;
        }
    }

    if (auto world = Comm::world().duplicated()) {
        Options opts;
        opts.parse({ argv, argv + argc });
        H1D::Driver{ std::move(world), opts }();
    } else {
        throw std::runtime_error{ std::string{ __PRETTY_FUNCTION__ } + " - invalid mpi::Comm" };
    }

    Comm::deinit();
    return 0;
} catch (...) {
    lippincott();
}
