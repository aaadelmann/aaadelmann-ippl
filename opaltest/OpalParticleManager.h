#ifndef OPALPARTICLEMANAGER_H
#define OPALPARTICLEMANAGER_H

/*
  Notes:

  1. need to switch solver without desiroying particle container

  2. field container do not make sense to me
   a) need to construct RHS i.e. \rho and J for the solver
   b) the solution(s) need to be backinterpolated to the particle in the PICmanager



 */

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

const char* TestName = "OPALTEST";

class OpalParticleManager
    : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>,
                              FieldSolver<double, 3>, LoadBalancer<double, 3>> {
    double loadbalancethreshold_m;
    double time_m;
    Vector_t<int, Dim> nr_m;
    size_type totalP_m;
    int nt_m;
    double lbt_m;
    double dt_m;
    int it_m;

    std::string step_method_m;
    std::string solver_m;

    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    Vector_t<double, Dim> length_m;
    Vector_t<double, Dim> hr_m;

    double Q;
    unsigned int nrMax_m;
    double dxFinest_m;

    bool isFirstRepartition;

    // Landau damping specific
    double Bext;
    double alpha;
    double DrInv;
    double rhoNorm_m;

    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;

public:
    OpalParticleManager(double totalCharge, Vector_t<int, Dim> nr, size_t totalP, int nt,
                        std::string solver, double lbt, std::string step_method)
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>,
                           FieldSolver<double, 3>, LoadBalancer<double, 3>>()
        , time_m(0.0)
        , nr_m(nr)
        , totalP_m(totalP)
        , nt_m(nt)
        , lbt_m(lbt)
        , dt_m(0)
        , it_m(0)
        , step_method_m(step_method)
        , solver_m(solver)
        , Q(totalCharge) {
        Inform m("OpalParticleManager() ");

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(nr[i]);
        }

        this->decomp_m.fill(true);  // all parallel

        this->rmin_m = 0.0;
        this->rmax_m = 20.0;

        this->length_m = this->rmax_m - this->rmin_m;
        this->hr_m     = this->length_m / this->nr_m;

        this->Bext     = 5.0;
        this->origin_m = this->rmin_m;

        this->nrMax_m    = 2048;  // Max grid size in our studies
        this->dxFinest_m = this->length_m[0] / this->nrMax_m;
        this->dt_m       = 0.5 * this->dxFinest_m;  // size of timestep

        this->alpha = -0.5 * this->dt_m;
        this->DrInv = 1.0 / (1 + (std::pow((this->alpha * this->Bext), 2)));

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        bool isAllPeriodic = true;

        Mesh_t<Dim>* mesh = new Mesh_t<Dim>(this->domain_m, this->hr_m, this->origin_m);
        FieldLayout_t<Dim>* FL =
            new FieldLayout_t<Dim>(MPI_COMM_WORLD, this->domain_m, this->decomp_m, isAllPeriodic);
        PLayout_t<T, Dim>* PL = new PLayout_t<T, Dim>(*FL, *mesh);

        this->pcontainer_m = std::make_shared<ParticleContainer_t>(*PL);
        this->fcontainer_m = std::make_shared<FieldContainer_t>(this->hr_m, this->rmin_m,
                                                                this->rmax_m, this->decomp_m);
        this->fcontainer_m->initializeFields(*mesh, *FL);

        this->fsolver_m = std::make_shared<FieldSolver_t>(
            this->solver_m, &this->fcontainer_m->rho_m, &this->fcontainer_m->E_m);
        this->fsolver_m->initSolver();
        this->loadbalancer_m = std::make_shared<LoadBalancer_t>(
            this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m);

        this->setParticleContainer(pcontainer_m);
        this->setFieldContainer(fcontainer_m);
        this->setFieldSolver(fsolver_m);
        this->setLoadBalancer(loadbalancer_m);

        this->initializeParticles(*mesh, *FL);
    }

    ~OpalParticleManager() {
        Inform m("OpalParticleManager Destructor ");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

public:
    void initFields() {
        Inform m("initFields ");
        this->fcontainer_m->rho_m = 0.0;
        this->fsolver_m->runSolver();
        this->par2grid();
        this->fsolver_m->runSolver();
        this->grid2par();
        m << "Done";
    }

    void initializeParticles(Mesh_t<Dim>& mesh_m, FieldLayout_t<Dim>& FL_m) {
        Inform m("Initialize Particles");

        Vector_t<double, Dim> mu, sd;
        for (unsigned d = 0; d < Dim; d++) {
            mu[d] = 0.5 * this->length_m[d] + this->origin_m[d];
        }
        sd[0] = 0.15 * this->length_m[0];
        sd[1] = 0.05 * this->length_m[1];
        sd[2] = 0.20 * this->length_m[2];

        using DistR_t = ippl::random::NormalDistribution<double, Dim>;
        // const double parR[2*Dim] = {mu[0], sd[0], mu[1], sd[1], mu[2], sd[2]};
        double* parR = new double[2 * Dim];
        parR[0]      = mu[0];
        parR[1]      = sd[0];
        parR[2]      = mu[1];
        parR[3]      = sd[1];
        parR[4]      = mu[2];
        parR[5]      = sd[2];
        DistR_t distR(parR);

        Vector_t<double, Dim> origin = this->origin_m;
        if ((this->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            this->isFirstRepartition       = true;
            const ippl::NDIndex<Dim>& lDom = FL_m.getLocalNDIndex();
            const int nghost               = this->fcontainer_m->rho_m.getNghost();
            auto rhoview                   = this->fcontainer_m->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            const auto hr = this->hr_m;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.full_pdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(&FL_m, &mesh_m);
            this->loadbalancer_m->repartition(&FL_m, &mesh_m, this->isFirstRepartition);
        }

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(FL_m, mesh_m);

        int seed        = 42;
        using size_type = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   DistR_t>;
        samplingR_t samplingR(distR, this->rmax_m, this->rmin_m, rlayout, this->totalP_m);
        size_type nlocal = samplingR.getLocalNum();

        this->pcontainer_m->create(nlocal);

        view_type* R_m = &this->pcontainer_m->R.getView();
        samplingR.generate(*R_m, rand_pool64);

        view_type* P_m = &this->pcontainer_m->P.getView();

        double muP[Dim] = {0.0, 0.0, 0.0};
        double sdP[Dim] = {1.0, 1.0, 1.0};
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P_m, rand_pool64, muP, sdP));

        Kokkos::fence();
        ippl::Comm->barrier();

        this->pcontainer_m->Q = this->Q / this->totalP_m;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->step_method_m == "LeapFrog") {
            LeapFrogStep();
        }
    }
    void LeapFrogStep() {
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        Inform m("LeapFrog");

        const double alpha_m = this->alpha;
        const double Bext_m  = this->Bext;
        const double DrInv_m = this->DrInv;
        const double V0      = 30 * this->length_m[2];

        const auto len = this->length_m;
        const auto ori = this->origin_m;

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        auto Rview = pc->R.getView();
        auto Pview = pc->P.getView();
        auto Eview = pc->E.getView();
        Kokkos::parallel_for(
            "Kick1", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                double Eext_x =
                    -(Rview(j)[0] - ori[0] - 0.5 * len[0]) * (V0 / (2 * Kokkos::pow(len[2], 2)));
                double Eext_y =
                    -(Rview(j)[1] - ori[1] - 0.5 * len[1]) * (V0 / (2 * Kokkos::pow(len[2], 2)));
                double Eext_z =
                    (Rview(j)[2] - ori[2] - 0.5 * len[2]) * (V0 / (Kokkos::pow(len[2], 2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];

                Pview(j)[0] += alpha_m * (Eext_x + Pview(j)[1] * Bext_m);
                Pview(j)[1] += alpha_m * (Eext_y - Pview(j)[0] * Bext_m);
                Pview(j)[2] += alpha_m * Eext_z;
            });
        Kokkos::fence();
        ippl::Comm->barrier();

        // drift
        pc->R = pc->R + dt_m * pc->P;

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        bool isFirstRepartition_m = false;
        if (loadbalancer_m->balance(this->totalP_m, this->it_m + 1)) {
            auto* mesh = &fc->rho_m.get_mesh();
            auto* FL   = &fc->getLayout();
            loadbalancer_m->repartition(FL, mesh, isFirstRepartition_m);
        }

        // scatter the charge onto the underlying grid
        this->par2grid();

        // Field solve
        this->fsolver_m->runSolver();

        // gather E field
        this->grid2par();

        auto R2view = pc->R.getView();
        auto P2view = pc->P.getView();
        auto E2view = pc->E.getView();

        Kokkos::parallel_for(
            "Kick2", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                double Eext_x = -(R2view(j)[0] - ori[0] - 0.5 * len[0])
                                * (V0 / (2 * Kokkos::pow(length_m[2], 2)));
                double Eext_y =
                    -(R2view(j)[1] - ori[1] - 0.5 * len[1]) * (V0 / (2 * Kokkos::pow(len[2], 2)));
                double Eext_z =
                    (R2view(j)[2] - ori[2] - 0.5 * len[2]) * (V0 / (Kokkos::pow(len[2], 2)));

                Eext_x += E2view(j)[0];
                Eext_y += E2view(j)[1];
                Eext_z += E2view(j)[2];

                P2view(j)[0] =
                    DrInv_m
                    * (P2view(j)[0]
                       + alpha_m * (Eext_x + P2view(j)[1] * Bext_m + alpha_m * Bext_m * Eext_y));
                P2view(j)[1] =
                    DrInv_m
                    * (P2view(j)[1]
                       + alpha_m * (Eext_y - P2view(j)[0] * Bext_m - alpha_m * Bext_m * Eext_x));
                P2view(j)[2] += alpha_m * Eext_z;
            });
        Kokkos::fence();
        ippl::Comm->barrier();
    }

    void par2grid() override { scatterCIC(); }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        Base::particle_position_type* E_p = &this->pcontainer_m->E;
        Base::particle_position_type* R_m = &this->pcontainer_m->R;
        VField_t<T, Dim>* E_f             = &this->fcontainer_m->E_m;
        gather(*E_p, *E_f, *R_m);
    }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->rho_m = 0.0;

        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        ippl::ParticleAttrib<double>* q_m = &this->pcontainer_m->Q;
        Base::particle_position_type* R_m = &this->pcontainer_m->R;
        Field_t<Dim>* rho_m               = &this->fcontainer_m->rho_m;
        const double Q_m                  = this->Q;

        scatter(*q_m, *rho_m, *R_m);

        m << std::fabs((Q_m - (*rho_m).sum()) / Q_m) << endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        ippl::Comm->reduce(local_particles, Total_particles, 1, std::plus<size_type>());

        double cellVolume =
            std::reduce(this->hr_m.begin(), this->hr_m.end(), 1., std::multiplies<double>());
        (*rho_m) = (*rho_m) / cellVolume;

        this->rhoNorm_m = norm(*rho_m);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= this->rmax_m[d] - this->rmin_m[d];
            }
            *rho_m = *rho_m - (Q_m / size);
        }
    }

    Inform& print(Inform& os) {
        // if (this->getLocalNum() != 0) {  // to suppress Nans
        Inform::FmtFlags_t ff = os.flags();

        os << std::scientific;
        os << level1 << "\n";
        os << "* ************** B U N C H "
              "********************************************************* \n";
        os << "* NP              = " << pcontainer_m->getLocalNum() << "\n";
        os << "* CORES           = " << ippl::Comm->size() << "\n";
        // os << "* THREADS         = " <<

        os << "* "
              "********************************************************************************"
              "** "
           << endl;
        os.flags(ff);
        // }
        return os;
    }
};

Inform& operator<<(Inform& os, OpalParticleManager& p) {
    return p.print(os);
}

#endif
