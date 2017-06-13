#include "approaches.h"
#include "bhtree.h"
#include "general.h"
#include "float3.h"
#include "body.h"

#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

#include <thread>
#include <future>


BHTree* build_tree(std::vector<Body*>& bodies)
{
    float
        min = std::numeric_limits<float>::infinity(),
        max = -std::numeric_limits<float>::infinity();
    for (const auto bi: bodies)
    {
        min = std::min(min, bi->position.x);
        min = std::min(min, bi->position.y);
        min = std::min(min, bi->position.z);
        max = std::max(max, bi->position.x);
        max = std::max(max, bi->position.y);
        max = std::max(max, bi->position.z);
    }
    float area_width = max-min+2;
    float3 start_pos = {min-1, min-1, min-1};

    // building the tree
    BHTree* btree = new BHTree(start_pos, area_width, LOCATION::ROOT);
    for(const auto bi: bodies)
    {
        btree->put_body(bi);
    }
    return btree;
}

namespace simulation
{

namespace sequential
{
    void nbody_nn_step(std::vector<Body*>& bodies, float delta)
    {
        const int n = bodies.size();
        float3 accelerations[n];

        int i = 0;
        float3 ai;
        for(const auto bi: bodies)
        {
            for(const auto bj: bodies)
                ai += body_body_iteraction(bi, bj);

            accelerations[i++] = ai;
            ai.clear();
        }

        // update positions, velocities, ceterka
        for(i = 0; i < n; i++)
        {
            bodies[i]->velocity += accelerations[i] * delta;
            bodies[i]->position += bodies[i]->velocity * delta;
        }
    }

    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta)
    {
        std::clock_t start, end;

        for(int i = 0; i < steps_count; i++)
        {
            std::cout << "[N body brute seq] " << i << " step\n";
            start = std::clock(); nbody_nn_step(bodies, delta); end = std::clock();
            std::cout << "Simulation step time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
        }    
    }

    void nbody_bh_step(std::vector<Body*>& bodies, BHTree* btree, float delta)
    {
        int i = 0;
        const int n = bodies.size();
        float3 accelerations[n], ai;
        for(const auto b: bodies)
        {
            accelerations[i++] = btree->get_acceleration_for_body(b);
            ai.clear();
        }

        // update positions, velocities, ceterka
        for(i = 0; i < n; i++)
        {
            bodies[i]->velocity += accelerations[i] * delta;
            bodies[i]->position += bodies[i]->velocity * delta;
        }
    }

    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta)
    {
        std::clock_t start, end;
        BHTree* btree;

        for(int i = 0; i < steps_count; i++)
        {
            std::cout << "[N body BH seq] " << i << " step\n";
            start = std::clock(); btree = build_tree(bodies); end = std::clock();
            std::cout << "Tree building time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
            start = std::clock(); nbody_bh_step(bodies, btree, delta); end = std::clock();
            std::cout << "Simulation step time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
            start = std::clock(); delete btree; end = std::clock();
            std::cout << "Tree deleting time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
        }
    }

    void simulate(int nbodies, int simulation_steps, float r_sphere, float min_m, float max_m)
    {
        std::cout << "---------------------------------------------------------------\n\n";
        std::clock_t    start, end;
        std::vector<Body*> bodies_a(nbodies), bodies_b(nbodies);
        
        float3 position, velocity;
        float m;
        start = std::clock();
        for(int i = 0; i < nbodies; i++)
        {
            position = get_onsphere_point(r_sphere);
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
        start = std::clock(); nbody_nn(bodies_a, simulation_steps); end = std::clock();
        std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
        for(auto bi: bodies_a)
        {
            delete bi;
        }

        // bh
        start = std::clock(); nbody_bh(bodies_b, simulation_steps); end = std::clock();
        std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
        for(auto bi: bodies_b)
        {
            delete bi;
        }
    }
}

namespace stdthread
{
    const int THREADS_COUNT = 2;
    void calculate_acceleration(std::vector<Body*>& bodies, std::vector<float3>& accelerations, int first, int last)
    {
        for(int i = first; i < last; i++)
        {
            for(const auto bj: bodies)
                accelerations[i] += body_body_iteraction(bodies[i], bj);
        }
    }

    void nbody_nn_step(std::vector<Body*>& bodies, float delta)
    {
        int n = bodies.size();
        std::vector<float3> accelerations(n);

        std::vector<std::future<void>> promises;
        int step_size = n/THREADS_COUNT;

        for(int i = 0; i < THREADS_COUNT; i++)
            promises.push_back(
                std::async(
                    std::launch::async,
                    calculate_acceleration, std::ref(bodies), std::ref(accelerations),
                    i*step_size, (i+1)*step_size
                )
            );

        for(auto& p: promises)
            p.get();

        // update positions, velocities, ceterka
        for(int i = 0; i < n; i++)
        {
            bodies[i]->velocity += accelerations[i] * delta;
            bodies[i]->position += bodies[i]->velocity * delta;
        }
    }

    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta)
    {
        std::clock_t start, end;

        for(int i = 0; i < steps_count; i++)
        {
            start = std::clock(); nbody_nn_step(bodies, delta); end = std::clock();
            std::cout <<"[nn]#" << i << " Simulation step time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
        }    
    }


    void nbody_bh_step(std::vector<Body*>& bodies, BHTree* btree, float delta)
    {
        int i = 0;
        const int n = bodies.size();
        float3 accelerations[n], ai;
        for(const auto b: bodies)
        {
            accelerations[i++] = btree->get_acceleration_for_body(b);
            ai.clear();
        }

        // update positions, velocities, ceterka
        for(i = 0; i < n; i++)
        {
            bodies[i]->velocity += accelerations[i] * delta;
            bodies[i]->position += bodies[i]->velocity * delta;
        }
    }

    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta)
    {
        std::clock_t start, end;
        BHTree* btree;

        for(int i = 0; i < steps_count; i++)
        {
            btree = build_tree(bodies);
            start = std::clock(); nbody_bh_step(bodies, btree, delta); end = std::clock();
            delete btree;
            std::cout << "[bh]#" << i << " Simulation step time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n";
        }
    }

    void simulate(int nbodies, int simulation_steps, float r_sphere, float min_m, float max_m)
    {
        std::cout << "---------------------------------------------------------------\n\n";
        std::clock_t    start, end;
        std::vector<Body*> bodies_a(nbodies), bodies_b(nbodies);
        
        float3 position, velocity;
        float m;
        start = std::clock();
        for(int i = 0; i < nbodies; i++)
        {
            position = get_onsphere_point(r_sphere);
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
        start = std::clock(); nbody_nn(bodies_a, simulation_steps); end = std::clock();
        std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
        for(auto bi: bodies_a)
        {
            delete bi;
        }

        // bh
        start = std::clock(); nbody_bh(bodies_b, simulation_steps); end = std::clock();
        std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
        for(auto bi: bodies_b)
        {
            delete bi;
        }
    }
}

}
