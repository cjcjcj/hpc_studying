#include "numerical.h"

#include <functional>
#include <cmath>
#include <vector>
#include <deque>

std::vector<float>
numerical::euler
(std::function<float(float, float)> f, float y0, float a, float b, float h)
{
    std::vector<float> rval;
    float y = y0;
    for (float t = a; t <= b; t += h)
    {
        rval.push_back(y);
        y += h * f(t, y);
    }
    return rval;
}

std::vector<float>
numerical::euler_cauchy
(std::function<float(float, float)> f, float y0, float a, float b, float h)
{
    std::vector<float> rval;
    float
        y = y0,
        yt;
    for (float t = a; t <= b; t += h)
    {
        rval.push_back(y);
        yt = y + h * f(t, y);
        y += h * .5 * (f(t, y) + f(t+h, yt)); 
    }
    return rval;
}

std::vector<float>
numerical::runge_kutta
(std::function<float(float, float)> f, float y0, float a, float b, float h)
{
    std::vector<float> rval;
    float
        y = y0,
        dy,
        k1, k2, k3, k4;
    k1 = k2 = k3 = k4 = 0;
    for (float t = a; t <= b; t += h)
    {
        dy = 1./6 * (k1 + 2*k2 + 2*k3 + k4);
        y += dy;
        rval.push_back(y);

        k1 = h*f(t, y);
        k2 = h*f(t + .5*h, y + .5*k1);
        k3 = h*f(t + .5*h, y + .5*k2);
        k4 = h*f(t + h, y + k3);
    }
    return rval;
}

std::vector<float>
numerical::adams
(std::function<float(float, float)> f, float y0, float a, float b, float h)
{
    std::vector<float> rval(runge_kutta(f, y0, a, a + 3*h, h));
    std::deque<float> adams_values(rval.begin(), rval.end());

    float y;

    for (float t = a + 4*h; t <= b; t += h)
    {
        y = adams_values[3] +
            h/24 * (
                55*f(t-h, adams_values[3]) - 
                59*f(t-2*h, adams_values[2]) +
                37*f(t-3*h, adams_values[1]) -
                9*f(t-4*h, adams_values[0])
            );
        rval.push_back(y);
        adams_values.push_back(y);
        adams_values.pop_front();
    }

    return rval;
}
