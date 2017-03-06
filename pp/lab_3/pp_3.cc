#include "approaches.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include <omp.h>

struct Intervals
{
    float a;
    float b;
};

void results2file(
    std::ofstream& results, std::string&& type,
    float e, const Intervals& i, std::pair<long, double>& r, double time)
{
    results << type << "," << e << "," << i.a << "," << i.b << ","
            << r.first << "," << time << std::endl;
}

void compare(
    std::function<double(const double)> f,
    const std::vector<Intervals>& intervals,
    const float e,
    const int num_iterations = 5,
    const std::string fname = "results.csv")
{
    std::ofstream results;
    results.open(fname);

    std::pair<long, double> r;
    double tstart, tend;

    for (const auto& el: intervals)
    {
        for(int i = 0; i < num_iterations; i++)
        {
            tstart = omp_get_wtime(),
            r = sequential(f, el.a, el.b, e),
            tend = omp_get_wtime();
            results2file(results, "sequential", e, el, r, tend-tstart);

            tstart = omp_get_wtime(),
            r = parallel_reduction(f, el.a, el.b, e),
            tend = omp_get_wtime();
            results2file(results, "parallel_reduction", e, el, r, tend-tstart);

            tstart = omp_get_wtime(),
            r = parallel_lock(f, el.a, el.b, e),
            tend = omp_get_wtime();
            results2file(results, "parallel_lock", e, el, r, tend-tstart);

            tstart = omp_get_wtime(),
            r = parallel_atomic(f, el.a, el.b, e),
            tend = omp_get_wtime();
            results2file(results, "parallel_atomic", e, el, r, tend-tstart);

            tstart = omp_get_wtime(),
            r = parallel_critical(f, el.a, el.b, e),
            tend = omp_get_wtime();
            results2file(results, "parallel_critical", e, el, r, tend-tstart);
        }
    }

    results.close();
}


int main()
{
    auto f = [] (const double x) -> double
        {return 1/std::pow(x, 2) * std::pow(std::sin(1/x), 2);};

    std::vector<Intervals> intervals = {
        {0.00001, 0.0001},
        {0.0001, 0.001},
        {0.001, 0.01},
        {0.01, 0.1},
        {0.1, 1},
        {1, 10},
    };

    float e = std::pow(10, -6);

    compare(f, intervals, e);
}
