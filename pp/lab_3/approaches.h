#pragma once

#include <functional>

std::pair<long, double> sequential(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e);

std::pair<long, double> parallel_reduction(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e);

std::pair<long, double> parallel_lock(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e);

std::pair<long, double> parallel_atomic(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e);

std::pair<long, double> parallel_critical(
    std::function<double(const double)> f,
    const double a, const double b,
    const float e);
