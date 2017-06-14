#include "float3.h"
#include "body.h"
#include "approaches.h"
#include "general.h"

#include <cmath>
#include <iostream>

int main()
{
    const int
        max_pow = 5,
        simulation_steps = 5,
        repeats = 5;

    // simulation::sequential::simulate(50000, 2);
    simulation::stdthread::simulate(10000, 2);
    return 1;

    for (int repeat = 0; repeat < repeats; repeat++)
    {
        std::cout << "cycle #" << repeat << std::endl;
        // for(int pow = 1; pow <= max_pow; pow++)
            // simulation::omp::simulate(std::pow(10, pow), simulation_steps);
        // simulation::omp::simulate(10000, 2);
        std::cout << std::endl;
    }
}
