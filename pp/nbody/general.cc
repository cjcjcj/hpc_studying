#include "float3.h"
#include "general.h"
#include "body.h"

#include <random>
#include <cmath>

std::random_device rd;
std::mt19937 gen(rd());

float rand(float r)
{
    std::uniform_real_distribution<float> dis(0, r);
    return dis(gen);
}
float rand(float l, float r)
{
    std::uniform_real_distribution<float> dis(l, r);
    return dis(gen);
}

float3 get_onsphere_point(float r)
{
    r -= .00001;
    const float
        phi = rand(PI_M2),
        sintheta = rand(-1.f, 1.f),
        costheta = std::sqrt(1.f - sintheta*sintheta);

    float3 point {
        r * std::cos(phi) * sintheta,
        r * std::sin(phi) * sintheta,
        r * costheta
    };
    return point;
}

// The softening factor models the interaction between two Plummer point masses:
// masses that behave as if they were spherical galaxies (Aarseth 2003, Dyer and Ip 1993).
float3 body_body_iteraction(const Body* bi, const Body* bj)
{
    float3 r;
    r = bj->position - bi->position;

    const float dsqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING_E*SOFTENING_E;
    const float almost_F = bj->m / std::pow(dsqr, 1.5);

    // bj.w * rij / distance^1.5
    r *= almost_F;

    return r;
}
