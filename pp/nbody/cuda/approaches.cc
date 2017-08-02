#include "approaches.h"
#include "float3.h"
#include "body.h"

#include <iostream>

#include <random>
#include <cmath>


std::random_device rd;
std::mt19937 gen(rd());

__host__ float rand(float r)
{
    std::uniform_real_distribution<float> dis(0, r);
    return dis(gen);
}
__host__ float rand(float l, float r)
{
    std::uniform_real_distribution<float> dis(l, r);
    return dis(gen);
}

__host__ float3 get_onsphere_point(float r)
{
    r -= .00001;
    const float
        theta = rand(PI_M2),
        u = rand(-1.f, 1.f),
        a = std::sqrt(1.f - u*u);

    float3 point {
        r * std::cos(theta) * a,
        r * std::sin(theta) * a,
        r * u
    };
    return point;
}

namespace simulation
{

__device__ float3 body_body_iteraction(const Body* bi, const Body* bj)
{
    float3 r;
    r = bj->position - bi->position;

    const float dsqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING_E*SOFTENING_E;
    const float almost_F = bj->m / std::pow(dsqr, 1.5);

    r *= almost_F;

    return r;
}

namespace cuda
{
    __device__ void calculate_acceleration(Body* bodies, float3* accelerations, int first, int last)
    {
        for(int i = first; i < last; i++)
        {
            for(const auto bj: bodies)
                accelerations[i] += body_body_iteraction(bodies[i], bj);
        }
    }

    __device__ void nbody_nn_step(Body* bodies, float delta)
    {
        int n = bodies.size();
        thrust::device_vector<float3> accelerations(n);

        // std::vector<std::thread> threads;
        // int step_size = n/THREADS_COUNT;

        // for(int i = 0; i < THREADS_COUNT; i++)
        //     threads.push_back(
        //         std::thread(
        //             calculate_acceleration, std::ref(bodies), std::ref(accelerations),
        //             i*step_size, (i+1)*step_size
        //         )
        //     );

        // for(auto& t: threads)
        //     t.join();

        // threads.clear();

        // int l, r;
        // for(int i = 0; i < THREADS_COUNT; i++)
        // {
        //     int l = i*step_size, r = (i+1)*step_size;
        //     threads.push_back(
        //         std::thread(
        //             [&accelerations, &bodies, l, r, delta]() -> void
        //             {
        //                 for(int i = l; i < r; i++)
        //                 {
        //                     bodies[i]->velocity += accelerations[i] * delta;
        //                     bodies[i]->position += bodies[i]->velocity * delta;
        //                 }
        //             }
        //         )
        //     );
        // }

        // for(auto& t: threads)
        //     t.join();
    }

    __device__ void nbody_nn(Body* bodies, int steps_count, float delta=1)
    {
        for(int i = 0; i < steps_count; i++)
        {
            nbody_nn_step(bodies, delta);
            // std::cout << "[nn]#" << i << " Simulation step time: " << time_span.count() << " s\n";
        }    
    }


    void simulate(int nbodies, int simulation_steps, float r_sphere, float min_m, float max_m)
    {
        std::cout << "---------------------------------------------------------------\n\n";

        // allocate memory
        // Body* bodies_host(nbodies);
        
        float3 position, velocity;
        float m;
        for(int i = 0; i < nbodies; i++)
        {
            position = get_onsphere_point(r_sphere);
            velocity = -position*.01f;
            m = rand(min_m, max_m);

            bodies_a[i] = Body(position, velocity, m);
        }
        std::cout << "bodies count: " << nbodies << std::endl;

        // allocate device memory + copy
        // Body bodies_device = bodies_host;
        // brute
        nbody_nn(bodies_device, simulation_steps);

        // // bh
        // start = std::chrono::high_resolution_clock::now(); nbody_bh(bodies_b, simulation_steps); end = std::chrono::high_resolution_clock::now();
        // std::cout << "Simulation time: " << std::chrono::duration<double>(end-start).count() << " s\n\n";
        // for(auto bi: bodies_b)
        // {
        //     delete bi;
        // }
    }
}

}
