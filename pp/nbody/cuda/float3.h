#pragma once
#include <iostream>

struct float3
{
    float x;
    float y;
    float z;

    __device__ __host__ float3 operator-();

    __device__ __host__ friend float3 operator-(float3, const float3&);
    __device__ __host__ friend float3 operator*(float3, float);
    __device__ __host__ friend float3 operator/(float3, float);

    __device__ __host__ float3& operator-=(const float3& rs);
    __device__ __host__ float3& operator+=(const float3& rs);
    __device__ __host__ float3& operator*=(float s);
    __device__ __host__ float3& operator/=(float s);
    __device__ __host__ bool operator==(const float3& rs) const;

    __device__ __host__ void clear();

};
