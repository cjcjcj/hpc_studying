#include "nbody.h"
#include <cmath>
#include <vector>

float3 operator-(float3 l, const float3& r)
{
    l += r;
    return l;
}

float3 operator*(float3 l, const float r)
{
    l *= r;
    return l;
}

// The softening factor models the interaction between two Plummer point masses:
// masses that behave as if they were spherical galaxies (Aarseth 2003, Dyer and Ip 1993).
float3 body_body_iteraction(const Body& bi, const Body& bj)
{
    float3 r;
    r = bj.position - bi.position;

    const float dsqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING_E*SOFTENING_E;
    const float almost_F = bj.m / std::pow(dsqr, 1.5);

    // bj.w * rij / distance^1.5
    r *= almost_F;

    return r;
}

// @delta -- time delta
void nbody_nn(std::vector<Body>& bodies, float delta)
{
    const int n = bodies.size();
    float3 accelerations[n];

    int i = 0;
    float3 ai;
    for(const auto& bi: bodies)
    {
        for(const auto& bj: bodies)
            ai += body_body_iteraction(bi, bj);


        accelerations[i++] = ai;
        ai.clear();
    }

    // update positions, velocities ceterka
    for(i = 0; i < n; i++)
    {
        bodies[i].velocity += accelerations[i] * delta;
        bodies[i].position += bodies[i].velocity * delta;
    }
}
