#pragma once
#include <vector>

__constant__ const float SOFTENING_E = 0.01f;

__constant__ const float PI          = 3.141592653589793;
__constant__ const float PI_M2       = 6.283185307179586;

namespace simulation
{

__device__ float3 body_body_iteraction(const Body& bi, const Body& bj);

namespace cuda
{
    __global__ void simulate(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f);
}

}

