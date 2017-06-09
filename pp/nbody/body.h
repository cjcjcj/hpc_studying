#pragma once

#include "float3.h"

struct Body
{
    Body();
    Body(float3 position, float3 velocity, float m);

    float3 position;
    float3 velocity;

    float m;

    bool operator==(const Body& rs)
    {
        return position == rs.position;
    }
};
