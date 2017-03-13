#include "nbody.h"

#include <vector>
#include <random>
#include <cmath>

#include <ctime>
#include <iostream>

#define PI      3.141592653589793f
#define PI_M2   6.283185307179586f

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

float3 getPoint(float r)
{
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

int main()
{
    std::clock_t    start, end;

    const int nbodies = 100000;
    std::vector<Body> bodies(nbodies);
    float
        r_sphere = 20,
        min_m = 1.f, max_m = 100.f;
    
    float3 position, velocity;
    float m;
    start = std::clock();
    for(int i = 0; i < nbodies; i++)
    {
        position = getPoint(r_sphere);
        velocity = -position*.01f;
        m = rand(min_m, max_m);

        bodies[i] = {position, velocity, m};
    }

    return 0;

    std::cout << "started\n";
    start = std::clock();
    nbody_nn(bodies);
    end = std::clock();
    std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " ms" << std::endl;
}