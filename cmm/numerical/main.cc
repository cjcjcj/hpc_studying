#include "numerical.h"

#include <iomanip>
#include <iostream>
#include <cmath>
#include <vector>

namespace utils
{
void diff_results
(
    const std::vector<float>& true_val,
    const std::vector<float>& method_results,
    float a, float h
)
{
    int last_ind = std::max(method_results.size(), true_val.size());
    for(int i = 0; i <= last_ind; i++)
        std::cout << std::fixed << std::setprecision(1)
                  << a + i*h << " "

                  << std::fixed << std::setprecision(9)
                  << method_results[i] << " "
                  << true_val[i] << " "

                  << std::scientific
                  << std::abs(method_results[i] - true_val[i])

                  << std::endl;
    std::cout << std::endl;
}

void show_method_result
(
    const std::vector<float>& method_results,
    float a, float h
)
{
    int size = method_results.size();
    for (int i = 0; i < size; i++)
        std::cout << std::fixed << std::setprecision(1)
                  << a + i*h << " "

                  << std::fixed << std::setprecision(9)
                  << method_results[i]
                  << std::endl;
    std::cout << std::endl;
}
}

int tasks()
{
    auto f = [] (float x, float y) -> float
        {return std::pow(y+x, 2);};
    auto f_true = [] (float x) -> float
        {return std::tan(x) - x;};

    float
        y0 = 0,
        a = 0,
        b = 1,
        h = .1;

    std::vector<float> f_true_values;
    for (float t = a; t <= b; t += h)
        f_true_values.push_back(f_true(t));

    std::vector<float> method_results;
    // euler
    std::cout << "euler\n";
    method_results = numerical::euler(f, y0, a, b, h);
    utils::diff_results(f_true_values, method_results, a, h);
    method_results.clear();

    // euler_cauchy
    std::cout << "euler_cauchy\n";
    method_results = numerical::euler_cauchy(f, y0, a, b, h);
    utils::diff_results(f_true_values, method_results, a, h);
    method_results.clear();

    // runge_kutta
    std::cout << "runge_kutta\n";
    method_results = numerical::runge_kutta(f, y0, a, b, h);
    utils::diff_results(f_true_values, method_results, a, h);
    method_results.clear();

    // adams
    std::cout << "adams\n";
    method_results = numerical::adams(f, y0, a, b, h);
    utils::diff_results(f_true_values, method_results, a, h);
    method_results.clear();

    return 0;
}

int modeling_task()
{
    auto law = [] (float , float x) -> float
        {return 1./3 * std::sqrt(x) * (8. - x);};

    float
        y0 = 1.5,
        a = 0,
        b = 10,
        h = 1;

    std::vector<float> method_results;
    // euler
    std::cout << "euler\n";
    method_results = numerical::euler(law, y0, a, b, h);
    utils::show_method_result(method_results, a, h);
    method_results.clear();

    // euler_cauchy
    std::cout << "euler_cauchy\n";
    method_results = numerical::euler_cauchy(law, y0, a, b, h);
    utils::show_method_result(method_results, a, h);
    method_results.clear();

    // runge_kutta
    std::cout << "runge_kutta\n";
    method_results = numerical::runge_kutta(law, y0, a, b, h);
    utils::show_method_result(method_results, a, h);
    method_results.clear();

    // adams
    std::cout << "adams\n";
    method_results = numerical::adams(law, y0, a, b, h);
    utils::show_method_result(method_results, a, h);
    method_results.clear();

    return 0;
}

int main(int argc, char const *argv[])
{
    modeling_task();
    return 0;
}