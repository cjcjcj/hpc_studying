#pragma once
#include <vector>
#include "body.h"
#include "bhtree.h"

void nbody_seq_step(std::vector<Body*>& bodies, float delta);
void nbody_seq(std::vector<Body*>& bodies, int steps_count, float delta=1);

void nbody_bh_seq_step(std::vector<Body*>& bodies, BHTree* btree, float delta);
BHTree* build_tree(std::vector<Body*>& bodies);
void nbody_bh_seq(std::vector<Body*>& bodies, int steps_count, float delta=1);

void sequential(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);