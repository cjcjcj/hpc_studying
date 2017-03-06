#pragma once

#include <functional>
#include <vector>

namespace numerical{
// https://web.archive.org/web/20130930101140/http://cpp-next.com/archive/2009/08/want-speed-pass-by-value
std::vector<float>
euler
(std::function<float(float, float)> f, float y0, float a, float b, float h);

std::vector<float>
euler_cauchy
(std::function<float(float, float)> f, float y0, float a, float b, float h);

std::vector<float>
runge_kutta
(std::function<float(float, float)> f, float y0, float a, float b, float h);

std::vector<float>
adams
(std::function<float(float, float)> f, float y0, float a, float b, float h);
}