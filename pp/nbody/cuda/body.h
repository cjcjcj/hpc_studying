#pragma once

#include "float3.h"

struct Body
{
    __device__ __host__ Body();
    __device__ __host__ Body(float3 position, float3 velocity, float m);

    float3 position;
    float3 velocity;

    float m;

    __device__ __host__ bool operator==(const Body& rs)
    {
        return position == rs.position;
    }
};
