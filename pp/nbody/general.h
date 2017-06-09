#pragma once
#include "float3.h"
#include "body.h"

const float SOFTENING_E = 0.01f;

#define PI      3.141592653589793f
#define PI_M2   6.283185307179586f

float rand(float r);
float rand(float l, float r);

float3 body_body_iteraction(const Body*, const Body*);
