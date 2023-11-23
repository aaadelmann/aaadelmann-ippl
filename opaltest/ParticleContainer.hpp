#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

public:
    ippl::ParticleAttrib<double> Q;              // charge
    ippl::ParticleAttrib<double> M;              // mass
    ippl::ParticleAttrib<double> dt;             // individual timestep
    ippl::ParticleAttrib<double> Phi;            // the scalar potential
    ippl::ParticleAttrib<short> Bin;             // the energy bin the particle is in
    typename Base::particle_position_type P;     // particle momenta
    typename Base::particle_position_type E;     // electric field at particle position
    typename Base::particle_position_type Etmp;  // electric field for gun simulation with bins
    typename Base::particle_position_type B;     // magnetic field at particle position

    ParticleContainer(ippl::ParticleSpatialLayout<T, Dim>& pl)
        : Base(pl) {
        this->initialize(pl);
        registerAttributes();
        setupBCs();
    }

    ~ParticleContainer() {}

    void registerAttributes() {
        // register the particle attributes
        this->addAttribute(Q);
        this->addAttribute(M);
        this->addAttribute(dt);
        this->addAttribute(Phi);
        this->addAttribute(Bin);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(Etmp);
        this->addAttribute(B);
    }
    void setupBCs() { setBCAllPeriodic(); }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
