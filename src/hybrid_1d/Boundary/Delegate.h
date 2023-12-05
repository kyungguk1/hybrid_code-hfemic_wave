/*
 * Copyright (c) 2019-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../Core/Domain.h"
#include "../ParamSet.h"
#include <PIC/GridArray.h>
#include <PIC/Particle.h>

#include <memory>
#include <vector>

HYBRID1D_BEGIN_NAMESPACE
class Delegate {
protected:
    struct BucketBuffer {
        using PartBucket = std::vector<Particle>;

        PartBucket L{};
        PartBucket R{};

        void swap_bucket() noexcept { L.swap(R); }
        void swap(BucketBuffer &other) noexcept
        {
            this->L.swap(other.L);
            this->R.swap(other.R);
        }
        [[nodiscard]] inline auto cleared() -> decltype(auto);

        explicit BucketBuffer()            = default;
        BucketBuffer(BucketBuffer const &) = delete;
        BucketBuffer &operator=(BucketBuffer const &) = delete;
    };

    [[nodiscard]] BucketBuffer &bucket_buffer() const;

private:
    std::unique_ptr<BucketBuffer> m_buckets; // be sure to clear the contents before use

public:
    Delegate &operator=(Delegate const &) = delete;
    Delegate(Delegate const &)            = delete;
    virtual ~Delegate();
    explicit Delegate() noexcept;

    // called once after initialization but right before entering loop
    //
    virtual void once(Domain &) const = 0;

    // called before and after every cycle of update
    //
    virtual void prologue(Domain const &, long inner_step_count) const = 0;
    virtual void epilogue(Domain const &, long inner_step_count) const = 0;

    // boundary value communication
    //
    virtual void partition(PartSpecies &, BucketBuffer &) const;
    virtual void boundary_pass(PartSpecies const &, BucketBuffer &) const;
    virtual void boundary_pass(Domain const &, PartSpecies &) const; // be aware of mutation of particle bucket occurring
    virtual void boundary_pass(Domain const &, ColdSpecies &) const = 0;
    virtual void boundary_pass(Domain const &, BField &) const      = 0;
    virtual void boundary_pass(Domain const &, EField &) const      = 0;
    virtual void boundary_pass(Domain const &, Charge &) const      = 0;
    virtual void boundary_pass(Domain const &, Current &) const     = 0;
    virtual void boundary_gather(Domain const &, Charge &) const    = 0;
    virtual void boundary_gather(Domain const &, Current &) const   = 0;
    virtual void boundary_gather(Domain const &, Species &) const   = 0;

private: // helpers
    template <class T, long N>
    static void pass(GridArray<T, N, Pad> &);
    template <class T, long N>
    static void gather(GridArray<T, N, Pad> &);
    static void periodic_particle_pass(PartSpecies const &, BucketBuffer &);
    static void reflecting_particle_pass(PartSpecies const &, BucketBuffer &);

    [[nodiscard]] static MFAVector randomize_gyrophase(MFAVector const &);
};
HYBRID1D_END_NAMESPACE
