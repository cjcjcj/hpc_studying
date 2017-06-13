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
#include <chrono>

#include <omp.h>


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

    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta=1)
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

    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta=1)
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
    const int THREADS_COUNT = std::thread::hardware_concurrency();
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
        std::cout << THREADS_COUNT;
        int n = bodies.size();
        std::vector<float3> accelerations(n);

        std::vector<std::thread> threads;
        int step_size = n/THREADS_COUNT;

        for(int i = 0; i < THREADS_COUNT; i++)
            threads.push_back(
                std::thread(
                    calculate_acceleration, std::ref(bodies), std::ref(accelerations),
                    i*step_size, (i+1)*step_size
                )
            );

        for(auto& t: threads)
            t.join();

        threads.clear();

        int l, r;
        for(int i = 0; i < THREADS_COUNT; i++)
        {
            int l = i*step_size, r = (i+1)*step_size;
            threads.push_back(
                std::thread(
                    [&accelerations, &bodies, l, r, delta]() -> void
                    {
                        for(int i = l; i < r; i++)
                        {
                            bodies[i]->velocity += accelerations[i] * delta;
                            bodies[i]->position += bodies[i]->velocity * delta;
                        }
                    }
                )
            );
        }

        for(auto& t: threads)
            t.join();
    }

    void nbody_bh_step(std::vector<Body*>& bodies, BHTree* btree, float delta)
    {
        const int n = bodies.size();
        std::vector<float3> accelerations(n);

        std::vector<std::thread> threads;
        int step_size = n/THREADS_COUNT;

        for(int i = 0; i < THREADS_COUNT; i++)
            threads.push_back(
                std::thread(
                    [&accelerations, &bodies, i, step_size, btree]() -> void
                    {
                        int l = i*step_size, r = (i+1)*step_size;
                        for(int j = l; j < r; j++)
                            accelerations[j] = btree->get_acceleration_for_body(bodies[j]);
                    }
                )
            );

        for(auto& t: threads)
            t.join();

        threads.clear();

        int l, r;
        for(int i = 0; i < THREADS_COUNT; i++)
        {
            int l = i*step_size, r = (i+1)*step_size;
            threads.push_back(
                std::thread(
                    [&accelerations, &bodies, l, r, delta]() -> void
                    {
                        for(int i = l; i < r; i++)
                        {
                            bodies[i]->velocity += accelerations[i] * delta;
                            bodies[i]->position += bodies[i]->velocity * delta;
                        }
                    }
                )
            );
        }

        for(auto& t: threads)
            t.join();
    }


    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta=1)
    {
        std::chrono::high_resolution_clock::time_point start, end;

        for(int i = 0; i < steps_count; i++)
        {
            start = std::chrono::high_resolution_clock::now(); nbody_nn_step(bodies, delta); end = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration<double>(end-start);
            std::cout << "[nn]#" << i << " Simulation step time: " << time_span.count() << " s\n";
        }    
    }

    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta=1)
    {
        std::chrono::high_resolution_clock::time_point start, end;
        BHTree* btree;

        for(int i = 0; i < steps_count; i++)
        {
            start = std::chrono::high_resolution_clock::now();
            btree = build_tree(bodies);
            nbody_bh_step(bodies, btree, delta);
            delete btree;
            end = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration<double>(end-start);
            std::cout << "[bh]#" << i << " Simulation step time: " << time_span.count() << " s\n";
        }
    }


    void simulate(int nbodies, int simulation_steps, float r_sphere, float min_m, float max_m)
    {
        std::cout << "---------------------------------------------------------------\n\n";
        std::chrono::high_resolution_clock::time_point start, end;
        std::vector<Body*> bodies_a(nbodies), bodies_b(nbodies);
        
        float3 position, velocity;
        float m;
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < nbodies; i++)
        {
            position = get_onsphere_point(r_sphere);
            velocity = -position*.01f;
            m = rand(min_m, max_m);

            bodies_a[i] = new Body(position, velocity, m);
            bodies_b[i] = new Body(position, velocity, m);
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout 
                  << "bodies count: " << nbodies << std::endl
                  << "initialization time: " << std::chrono::duration<double>(end-start).count() << " s\n\n";


        // brute
        start = std::chrono::high_resolution_clock::now(); nbody_nn(bodies_a, simulation_steps); end = std::chrono::high_resolution_clock::now();
        std::cout << "Simulation time: " << std::chrono::duration<double>(end-start).count() << " s\n\n";
        for(auto bi: bodies_a)
        {
            delete bi;
        }

        // bh
        start = std::chrono::high_resolution_clock::now(); nbody_bh(bodies_b, simulation_steps); end = std::chrono::high_resolution_clock::now();
        std::cout << "Simulation time: " << std::chrono::duration<double>(end-start).count() << " s\n\n";
        for(auto bi: bodies_b)
        {
            delete bi;
        }
    }
}

namespace omp
{
    void nbody_nn_step(std::vector<Body*>& bodies, float delta)
    {
        const int n = bodies.size();
        float3 accelerations[n];

        float3 ai;
        #pragma omp parallel shared(bodies, accelerations)
        {
            #pragma omp for private(ai)
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                    if (i != j)
                        ai += body_body_iteraction(bodies.at(i), bodies.at(j));

                accelerations[i] = ai;
                ai.clear();
            }

            #pragma omp for
            for(int i = 0; i < n; i++)
            {
                bodies[i]->velocity += accelerations[i] * delta;
                bodies[i]->position += bodies[i]->velocity * delta;
            }
        }
    }

    void nbody_nn(std::vector<Body*>& bodies, int steps_count, float delta=1)
    {
        double start, end;

        for(int i = 0; i < steps_count; i++)
        {
            start = omp_get_wtime(); nbody_nn_step(bodies, delta); end = omp_get_wtime();
            std::cout <<"[nn]#" << i << " Simulation step time: " << end - start << " s\n";
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

    void nbody_bh(std::vector<Body*>& bodies, int steps_count, float delta=1)
    {
        double start, end;
        BHTree* btree;

        for(int i = 0; i < steps_count; i++)
        {
            btree = build_tree(bodies);
            start = omp_get_wtime(); nbody_bh_step(bodies, btree, delta); end = omp_get_wtime();
            delete btree;
            std::cout << "[bh]#" << i << " Simulation step time: " << end-start << " s\n";
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
        double t1, t2;
        t1=omp_get_wtime(); nbody_nn(bodies_a, simulation_steps); t2=omp_get_wtime();
        std::cout << "Simulation time: " << t2-t1 << " s\n\n";
        for(auto bi: bodies_a)
        {
            delete bi;
        }

        // // bh
        // start = std::clock(); nbody_bh(bodies_b, simulation_steps); end = std::clock();
        // std::cout << "Simulation time: " << (end - start) / (double)(CLOCKS_PER_SEC) << " s\n\n";
        // for(auto bi: bodies_b)
        // {
        //     delete bi;
        // }
    }
}

}
