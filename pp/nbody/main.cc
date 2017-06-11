#include "float3.h"
#include "body.h"
#include "sequential.h"
#include "general.h"

#include <cmath>
#include <iostream>

int main()
{
    const int
        max_pow = 5,
        simulation_steps = 5,
        repeats = 5;

    for (int repeat = 0; repeat < repeats; repeat++)
    {
        std::cout << "cycle #" << repeat << std::endl;
        for(int pow = 1; pow <= max_pow; pow++)
            sequential(std::pow(10, pow), simulation_steps);
        std::cout << std::endl;
    }
}
