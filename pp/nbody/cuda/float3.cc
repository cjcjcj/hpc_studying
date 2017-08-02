#include "float3.h"

__device__ __host__ float3& float3::operator+=(const float3& rs)
{
    this->x += rs.x; this->y += rs.y; this->z += rs.z;
    return *this;
}

__device__ __host__ float3& float3::operator-=(const float3& rs)
{
    this->x -= rs.x; this->y -= rs.y; this->z -= rs.z;
    return *this;
}

__device__ __host__ float3& float3::operator*=(const float s)
{
    this->x *= s; this->y *= s; this->z *= s;
    return *this;
}

__device__ __host__ float3& float3::operator/=(const float s)
{
    this->x /= s; this->y /= s; this->z /= s;
    return *this;
}

__device__ __host__ bool float3::operator==(const float3& rs) const
{
    return (this->x == rs.x) and (this->y == rs.y) and (this->z == rs.z);
}

__device__ __host__ float3 float3::operator-()
{ 
    return float3 {-x, -y, -z};
}

__device__ __host__ void float3::clear()
{
    x = y = z = 0;
}

__device__ __host__ float3 operator-(float3 l, const float3& r)
{
    l -= r;
    return l;
}

__device__ __host__ float3 operator/(float3 l, float r)
{
    l /= r;
    return l;
}

__device__ __host__ float3 operator*(float3 l, float r)
{
    l *= r;
    return l;
}
