#include "bhtree.h"
#include "body.h"
#include "float3.h"
#include "general.h"

#include <iostream>
#include <cmath>

Node::Node()
{}

Node::Node(float3 start_pos, float width, float3 position={.0, .0, .0}, float m=0)
{
    this->start_pos = start_pos;
    this->width = width;
    this->position = position;
    this->m = m;
}


InternalNode::InternalNode(float3 start_pos, float width)
: Node(start_pos, width)
{
    position = {.0, .0, .0};
    m = .0;
}

void Node::update(const Body* b)
{
    float _m = m + b->m;

    position.x = (position.x*m + b->position.x*b->m)/_m;
    position.y = (position.y*m + b->position.y*b->m)/_m;
    position.z = (position.z*m + b->position.z*b->m)/_m;

    m = _m;
}

void ExternalNode::update(const Body* b)
{
    position = b->position;
    m = b->m;
    body = b;
}


ExternalNode::ExternalNode(float3 start_pos, float width, const Body* body)
: Node(start_pos, width)
{
    if (body != nullptr)
    {
        this->position = body->position;
        this->m = body->m;
    }
    else
    {
        this->position = start_pos;
        this->m = .0;
    }
    this->body = body;
}


BHTree::BHTree(float3& start_pos, float width, LOCATION loc=LOCATION::ROOT)
{
    node = new ExternalNode(start_pos, width, nullptr);
    node_type = NodeType::EXTERNAL;
    location = loc;
}

BHTree::~BHTree()
{
    if (node_type==NodeType::INTERNAL)
        for(int i = 0; i < 8; i++)
            delete leafs[i];
    delete node;
}

int BHTree::which_oct(const Body* body)
{
    bool east, south, down;
    east  = (body->position.x - node->start_pos.x) >= node->width/2;
    south = (body->position.y - node->start_pos.y) >= node->width/2;
    down  = (body->position.z - node->start_pos.z) >= node->width/2;

    int loc = 0;
    if (east and south)
        loc = int(LOCATION::USE);
    else if(east and not south)
        loc = int(LOCATION::UNE);
    else
        loc = int(LOCATION::UNW);

    if (down)
        loc += 4;

    return loc;
}

void BHTree::put_body(const Body* body)
{
    if (node_type == NodeType::EXTERNAL)
    {
        ExternalNode* _curnode = static_cast<ExternalNode*>(node);
        if (_curnode->body == nullptr)
        {
            _curnode->update(body);
            return;
        }
        else
        {
            const Body* old_body = _curnode->body;
            init_subtree();
            node = new InternalNode(_curnode->start_pos, _curnode->width);
            node_type = NodeType::INTERNAL;
            delete _curnode;

            if (old_body != nullptr and old_body != body)
            {
                node->update(old_body);
                int loc = which_oct(old_body);
                leafs[loc]->put_body(old_body);
            }
        }
    }

    node->update(body);
    int loc = which_oct(body);
    leafs[loc]->put_body(body);
}

void BHTree::init_subtree()
{
    for(int i=0; i<8; i++)
    {
        float3 node_pos = node->start_pos;
        float node_w = node->width / 2.;

        if (i != 0)
        {
            if (i % 2 != 0)
                // east
                node_pos.x += node_w;

            if ((i > 1 and i < 4) or (i > 5))
                // south
                node_pos.y += node_w;

            if (i >= 4)
                // down octs
                node_pos.z += node_w;
        }

        leafs[i] = new BHTree(node_pos, node_w, LOCATION(i));
    }
}

float3 BHTree::get_acceleration_for_body(const Body* b, float theta)
{
    float3 acceleration = {.0, .0, .0};
    if (node_type == NodeType::INTERNAL)
    {
        float3 d = this->node->position - b->position;
        float distance;
        distance = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);

        if (node->width/distance < theta)
        {
            Body tmp = Body(this->node->position, float3{.0,.0,.0}, node->m);
            acceleration += body_body_iteraction(b, &tmp);
        }
        else
            for(int i = 0; i < 8; i++)
                acceleration += leafs[i]->get_acceleration_for_body(b, .1);
    }
    else
    {
        const Body* node_body = static_cast<ExternalNode*>(node)->body;
        if (node_body != nullptr)
            acceleration += body_body_iteraction(b, node_body);
    }
    return acceleration;
}