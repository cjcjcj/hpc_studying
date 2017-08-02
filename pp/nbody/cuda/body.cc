#include "body.h"
#include "float3.h"

__device__ __host__ Body::Body()
{
    position = {.0, .0, .0};
    velocity = {.0, .0, .0};
    m = .0;
}

__device__ __host__ Body::Body(float3 position, float3 velocity, float m)
{
    this->position = position;
    this->velocity = velocity;
    this->m = m;
}
