/*
 * Copyright (c) 2021-2022, Kyungguk Min
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "TestParticleVDF.h"

LIBPIC_NAMESPACE_BEGIN(1)
auto RelativisticTestParticleVDF::impl_emit(Badge<Super>, unsigned long const n) const -> std::vector<Particle>
{
    std::vector<Particle> ptls(n);
    std::generate(begin(ptls), end(ptls), [this] {
        return this->emit();
    });
    return ptls;
}
auto RelativisticTestParticleVDF::impl_emit(Badge<Super>) const -> Particle
{
    Particle ptl = load();
    {
        ptl.psd = { 0, 0, 1 };
    }
    return ptl;
}
auto RelativisticTestParticleVDF::load() const -> Particle
{
    if (m_particles.empty())
        return {}; // this assumes that the default-constructed particle object is not consumed by callers

    auto ptl = m_particles.back();
    m_particles.pop_back();
    return ptl;
}
LIBPIC_NAMESPACE_END(1)
