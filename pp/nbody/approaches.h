#pragma once
#include <vector>
#include "body.h"
#include "bhtree.h"


BHTree* build_tree(std::vector<Body*>& bodies);

namespace simulation
{

namespace sequential
{
    void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}

namespace stdthread
{
    void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}

namespace omp
{
    void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}


}

