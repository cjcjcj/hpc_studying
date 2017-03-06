#include "approaches.h"

#include <limits>
#include <cmath>

#include <omp.h>

std::pair<long, double> sequential(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e)
{
    double 
        prev_iter = std::numeric_limits<double>::infinity(),
        cur_iter = std::numeric_limits<double>::infinity(),
        step = std::numeric_limits<double>::infinity(),
        result = .0;
    long n = 100;

    while (
            std::isinf(prev_iter) ||
            std::abs(cur_iter - prev_iter) > e*std::abs(cur_iter)
        )
    {
        result = .0;
        n *= 2;

        step = (b-a)/n;
        for(long i = 1; i < n-1; i++)
            result += f(a + i*step);
        result = step / 2 * (f(a) + 2*result + f(b));

        prev_iter = cur_iter;
        cur_iter = result;
    }

    return {n, cur_iter};
}


std::pair<long, double> parallel_reduction(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e)
{
    double 
        prev_iter = std::numeric_limits<double>::infinity(),
        cur_iter = std::numeric_limits<double>::infinity(),
        step = std::numeric_limits<double>::infinity(),
        result = .0;
    long n = 100;

    while (
            std::isinf(prev_iter) ||
            std::abs(cur_iter - prev_iter) > e*std::abs(cur_iter)
        )
    {
        result = .0;
        n *= 2;

        step = (b-a)/n;
        #pragma omp parallel for reduction(+: result)
        for(long i = 1; i < n-1; i++)
            result += f(a + i*step);
        result = step / 2 * (f(a) + 2*result + f(b));

        prev_iter = cur_iter;
        cur_iter = result;
    }

    return {n, cur_iter};
}

std::pair<long, double> parallel_lock(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e)
{
    omp_lock_t my_lock;
    omp_init_lock(&my_lock);

    double 
        prev_iter = std::numeric_limits<double>::infinity(),
        cur_iter = std::numeric_limits<double>::infinity(),
        step = std::numeric_limits<double>::infinity(),
        result = .0,
        local_result = .0;
    long n = 100;

    while (
            std::isinf(prev_iter) ||
            std::abs(cur_iter - prev_iter) > e*std::abs(cur_iter)
        )
    {
        result = .0;
        n *= 2;

        step = (b-a)/n;
        #pragma omp parallel for private(local_result)
        for(long i = 1; i < n-1; i++)
        {
            local_result = f(a + i*step);

            omp_set_lock(&my_lock);
            result += local_result;
            omp_unset_lock(&my_lock);
        }
        result = step / 2 * (f(a) + 2*result + f(b));

        prev_iter = cur_iter;
        cur_iter = result;
    }

    omp_destroy_lock(&my_lock);

    return {n, cur_iter};
}

std::pair<long, double> parallel_atomic(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e)
{
    double 
        prev_iter = std::numeric_limits<double>::infinity(),
        cur_iter = std::numeric_limits<double>::infinity(),
        step = std::numeric_limits<double>::infinity(),
        result = .0;
    long n = 100;

    while (
            std::isinf(prev_iter) ||
            std::abs(cur_iter - prev_iter) > e*std::abs(cur_iter)
        )
    {
        result = .0;
        n *= 2;

        step = (b-a)/n;
        #pragma omp parallel for
        for(long i = 1; i < n-1; i++)
        {
            #pragma omp atomic
            result += f(a + i*step);
        }
        result = step / 2 * (f(a) + 2*result + f(b));

        prev_iter = cur_iter;
        cur_iter = result;
    }

    return {n, cur_iter};
}

std::pair<long, double> parallel_critical(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e)
{
    double 
        prev_iter = std::numeric_limits<double>::infinity(),
        cur_iter = std::numeric_limits<double>::infinity(),
        step = std::numeric_limits<double>::infinity(),
        result = .0,
        local_result = .0;
    long n = 100;

    while (
            std::isinf(prev_iter) ||
            std::abs(cur_iter - prev_iter) > e*std::abs(cur_iter)
        )
    {
        result = .0;
        n *= 2;

        step = (b-a)/n;
        #pragma omp parallel for private(local_result)
        for(long i = 1; i < n-1; i++)
        {
            local_result = f(a + i*step);
            #pragma omp critical
            result += local_result;
        }
        result = step / 2 * (f(a) + 2*result + f(b));

        prev_iter = cur_iter;
        cur_iter = result;
    }

    return {n, cur_iter};
}
