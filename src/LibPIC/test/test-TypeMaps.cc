/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "catch2/catch.hpp"

#define LIBPIC_INLINE_VERSION 1
#include <PIC/TypeMaps.h>
#include <memory>
#include <string>

TEST_CASE("Test LibPIC::TypeMaps::ParallelKit", "[LibPIC::TypeMaps::ParallelKit]")
{
    using parallel::make_type;

    try {
        using T      = Scalar;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = Vector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CartVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = MFAVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CovarVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = ContrVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourCartVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourMFAVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourCovarVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourContrVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = Tensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CartTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = MFATensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CovarTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = ContrTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourCartTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourMFATensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourCovarTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = FourContrTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CartCoord;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = CurviCoord;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = Particle;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }

    try {
        using T      = RelativisticParticle;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.alignment() == alignof(T));
        CHECK(t.signature_size() == sizeof(T));

        auto [lb, extent] = t.extent();
        CHECK(lb == 0);
        CHECK(extent == sizeof(T));

        std::tie(lb, extent) = t.true_extent();
        CHECK(lb == 0);
        REQUIRE(extent == sizeof(T));
    } catch (std::exception const &e) {
        INFO(e.what())
        CHECK(false);
    }
}
TEST_CASE("Test LibPIC::TypeMaps::HDF5Kit", "[LibPIC::TypeMaps::HDF5Kit]")
{
    using hdf5::make_type;

    try {
        using T      = Scalar;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = Vector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CartVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = MFAVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CovarVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = ContrVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourCartVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourMFAVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourCovarVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourContrVector;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = Tensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CartTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = MFATensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CovarTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = ContrTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourCartTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourMFATensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourCovarTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = FourContrTensor;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CartCoord;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = CurviCoord;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_ARRAY);
        CHECK(H5Tequal(*t.super_(), H5T_NATIVE_DOUBLE));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = Particle;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_COMPOUND);

        auto const n_members = H5Tget_nmembers(*t);
        REQUIRE(n_members == 4);

        using char_ptr = std::unique_ptr<char, void (*)(void *)>;
        char_ptr name{ nullptr, &free };

        CHECK(H5Tget_member_class(*t, 0) == H5T_ARRAY); // vel
        name = char_ptr{ H5Tget_member_name(*t, 0), &free };
        CHECK((!!name && std::string{ "vel" } == name.get()));

        CHECK(H5Tget_member_class(*t, 1) == H5T_ARRAY); // pos
        name = char_ptr{ H5Tget_member_name(*t, 1), &free };
        CHECK((!!name && std::string{ "pos" } == name.get()));

        CHECK(H5Tget_member_class(*t, 2) == H5T_ARRAY); // psd
        name = char_ptr{ H5Tget_member_name(*t, 2), &free };
        CHECK((!!name && std::string{ "psd" } == name.get()));

        CHECK(H5Tget_member_class(*t, 3) == H5T_INTEGER); // id
        name = char_ptr{ H5Tget_member_name(*t, 3), &free };
        CHECK((!!name && std::string{ "id" } == name.get()));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }

    try {
        using T      = RelativisticParticle;
        auto const t = make_type<T>();
        REQUIRE(!!t);
        CHECK(t.size() == sizeof(T));
        CHECK(t.class_() == H5T_COMPOUND);

        auto const n_members = H5Tget_nmembers(*t);
        REQUIRE(n_members == 4);

        using char_ptr = std::unique_ptr<char, void (*)(void *)>;
        char_ptr name{ nullptr, &free };

        CHECK(H5Tget_member_class(*t, 0) == H5T_ARRAY); // vel
        name = char_ptr{ H5Tget_member_name(*t, 0), &free };
        CHECK((!!name && std::string{ "gcgvel" } == name.get()));

        CHECK(H5Tget_member_class(*t, 1) == H5T_ARRAY); // pos
        name = char_ptr{ H5Tget_member_name(*t, 1), &free };
        CHECK((!!name && std::string{ "pos" } == name.get()));

        CHECK(H5Tget_member_class(*t, 2) == H5T_ARRAY); // psd
        name = char_ptr{ H5Tget_member_name(*t, 2), &free };
        CHECK((!!name && std::string{ "psd" } == name.get()));

        CHECK(H5Tget_member_class(*t, 3) == H5T_INTEGER); // id
        name = char_ptr{ H5Tget_member_name(*t, 3), &free };
        CHECK((!!name && std::string{ "id" } == name.get()));
    } catch (std::exception const &e) {
        INFO("Exception thrown: " << e.what());
        REQUIRE(false);
    }
}
