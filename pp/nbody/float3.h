#pragma once
#include <iostream>

struct float3
{
    float x;
    float y;
    float z;

    float3 operator-();

    friend float3 operator-(float3, const float3&);
    friend float3 operator*(float3, float);
    friend float3 operator/(float3, float);

    float3& operator-=(const float3& rs);
    float3& operator+=(const float3& rs);
    float3& operator*=(float s);
    float3& operator/=(float s);
    bool operator==(const float3& rs) const;

    void clear();

    friend std::ostream& operator<< (std::ostream& os, const float3& f)
    {
        os  << "{" 
            << f.x << ", "
            << f.y << ", "
            << f.z
            << "}";
        return os;
    }
};
