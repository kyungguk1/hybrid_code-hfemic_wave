/*
 * Copyright (c) 2021, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include <ParallelKit/ParallelKit.h>

int main(int argc, char *argv[])
{
    using parallel::mpi::Comm;

    // global setup...
    Comm::init(&argc, &argv);

    int result = Catch::Session().run(argc, argv);

    // global clean-up...
    Comm::deinit();

    return result;
}
