#pragma once
#include <vector>

const float SOFTENING_E = 0.01f;

struct float3
{
    float x;
    float y;
    float z;

    friend float3 operator-(float3, const float3&);
    friend float3 operator*(float3, const float);

    float3& operator+=(const float3& rs)
    {
        this->x += rs.x; this->y += rs.y; this->z += rs.z;
        return *this;
    }

    float3& operator*=(const float s)
    {
        this->x *= s; this->y *= s; this->z *= s;
        return *this;
    }

    float3 operator-()
    {
        return float3 {-x, -y, -z};
    }

    void clear()
    {
        x = y = z = 0;
    }

};

struct Body
{
    float3 position;
    float3 velocity;

    float m;
};

float3 body_body_iteraction(const Body&, const Body&);
void nbody_nn(std::vector<Body>& bodies, float delta=1);