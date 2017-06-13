#pragma once
#include <vector>
#include "body.h"
#include "bhtree.h"


BHTree* build_tree(std::vector<Body*>& bodies);

namespace simulation
{

namespace sequential
{
    void nbody_nn_step(std::vector<Body*>& bodies, float delta);
    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta=1);

    void nbody_bh_step(std::vector<Body*>& bodies, BHTree* btree, float delta);
    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta=1);

    void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}

namespace stdthread
{
    void nbody_nn_step(std::vector<Body*>& bodies, float delta);
    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta=1);

    void nbody_bh_step(std::vector<Body*>& bodies, BHTree* btree, float delta);
    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta=1);

    void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}

}

