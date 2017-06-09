#include "float3.h"

float3& float3::operator+=(const float3& rs)
{
    this->x += rs.x; this->y += rs.y; this->z += rs.z;
    return *this;
}

float3& float3::operator-=(const float3& rs)
{
    this->x -= rs.x; this->y -= rs.y; this->z -= rs.z;
    return *this;
}

float3& float3::operator*=(const float s)
{
    this->x *= s; this->y *= s; this->z *= s;
    return *this;
}

float3& float3::operator/=(const float s)
{
    this->x /= s; this->y /= s; this->z /= s;
    return *this;
}

bool float3::operator==(const float3& rs) const
{
    return (this->x == rs.x) and (this->y == rs.y) and (this->z == rs.z);
}


float3 float3::operator-()
{
    return float3 {-x, -y, -z};
}

void float3::clear()
{
    x = y = z = 0;
}

float3 operator-(float3 l, const float3& r)
{
    l -= r;
    return l;
}

float3 operator/(float3 l, float r)
{
    l /= r;
    return l;
}

float3 operator*(float3 l, float r)
{
    l *= r;
    return l;
}
