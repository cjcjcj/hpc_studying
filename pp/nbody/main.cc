#include "float3.h"
#include "body.h"
#include "approaches.h"
#include "general.h"

#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>


float3 getPoint(float r)
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

void sequential(int nbodies, int simulation_steps, float r_sphere=4000, float min_m=1.f, float max_m=100.f)
{
    std::cout << "---------------------------------------------------------------\n\n";
    std::clock_t    start, end;
    std::vector<Body*> bodies_a(nbodies), bodies_b(nbodies);
    
    float3 position, velocity;
    float m;
    start = std::clock();
    for(int i = 0; i < nbodies; i++)
    {
        position = getPoint(r_sphere);
        velocity = -position*.01f;
        m = rand(min_m, max_m);

        bodies_a[i] = new Body(position, velocity, m);
        bodies_b[i] = new Body(position, velocity, m);
    }
    end = std::clock();
    std::cout 
              << "bodies count: " << nbodies << std::endl
              << "initialization time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";

    // brute
    start = std::clock(); nbody_seq(bodies_a, simulation_steps); end = std::clock();
    std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
    for(auto bi: bodies_a)
    {
        delete bi;
    }

    // bh
    start = std::clock(); nbody_bh_seq(bodies_b, simulation_steps); end = std::clock();
    std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
    for(auto bi: bodies_b)
    {
        delete bi;
    }
}

int main()
{
    const int
        simulation_steps = 5,
        max_pow = 3,
        repeats = 5;

    for (int repeat = 0; repeat < repeats; repeat++)
    {
        std::cout << "cycle #" << repeat << std::endl;
        for(int pow=1; pow <= max_pow; pow++)
            sequential(std::pow(10, pow), simulation_steps);
        std::cout << std::endl;
    }
}
