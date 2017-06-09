#pragma once
#include "float3.h"
#include "body.h"

#include <iostream>

struct Node
{    
    float3 start_pos;
    float width;
    float3 position;
    float m;

    virtual void update(const Body* b);

    Node();
    Node(float3 start_pos, float width, float3 position, float m);
};

struct InternalNode : public Node
{
    InternalNode(float3 start_pos, float width);
};

struct ExternalNode : public Node
{
    const Body* body;

    ExternalNode(float3 start_pos, float width, const Body* body);
    void update(const Body* b);
};

enum LOCATION {
        ROOT=-1,
        UNW, UNE, USW, USE,
        DNW, DNE, DSW, DSE
    };

class BHTree
{
private:
    LOCATION location;
    enum NodeType {
        INTERNAL, EXTERNAL
    } node_type;

    Node* node;
    BHTree* leafs[8];
public:
    BHTree(float3& start_pos, float width, LOCATION loc);
    ~BHTree();

    void put_body(const Body* body);
    friend std::ostream& operator<< (std::ostream& os, const BHTree& bht);
    float3 get_acceleration_for_body(const Body* b, float theta=.5);
private:
    void init_subtree();
    int which_oct(const Body* body);
};
