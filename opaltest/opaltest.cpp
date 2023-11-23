// opaltest
//   Usage:
//     srun ./opaltest
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny       = No. cell-centered points in the y-direction
//     nz       = No. cell-centered points in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT, CG, P3M, and OPEN supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./opaltest 128 128 128 10000 300 FFT 0.01 LeapFrog --overallocate 1.0 --info 10
//
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "OpalParticleManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("OpalTest");
        Inform msg2all("OpalTest", INFORM_ALL_NODES);

        // Read input parameters, assign them to the corresponding memebers of manager
        int arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        size_t totalP           = std::atoll(argv[arg++]);
        int nt                  = std::atoi(argv[arg++]);
        std::string solver      = argv[arg++];
        double lbt              = std::atof(argv[arg++]);
        std::string step_method = argv[arg++];

        double QTot = -1562.5;

        OpalParticleManager manager(QTot, nr, totalP, nt, solver, lbt, step_method);

        // Perform pre-run operations, including creating mesh, particles,...
        manager.initFields();

        msg << "Starting iterations ..." << endl;

        manager.run(nt);

        msg << manager << endl;

        msg << "End." << endl;
    }
    ippl::finalize();
    return 0;
}
