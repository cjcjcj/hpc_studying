#include "sendfunctions.h"
#include "constants.h"

#include "mpi.h"

#include <string>

double Send(std::string& msg, int msg_tag)
{
    double start;

    start = MPI::Wtime();
    MPI::COMM_WORLD.Send(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return MPI::Wtime() - start;
}

double Ssend(std::string& msg, int msg_tag)
{
    double start;

    start = MPI::Wtime();
    MPI::COMM_WORLD.Ssend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return MPI::Wtime() - start;
}

double Bsend(std::string& msg, int msg_tag)
{
    double start, end;
    void * buf = new char[msg.length()];

    MPI::Attach_buffer(buf, msg.size() + MPI::BSEND_OVERHEAD);
    start = MPI::Wtime();
    MPI::COMM_WORLD.Bsend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);
    end = MPI::Wtime();
    MPI::Detach_buffer(buf);

    return end - start;
}

double Rsend(std::string& msg, int msg_tag)
{
    double start;

    start = MPI::Wtime();
    MPI::COMM_WORLD.Rsend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return MPI::Wtime() - start;
}

std::pair<double, MPI::Request> ISend(std::string& msg, int msg_tag)
{
    double start;
    MPI::Request request;

    start = MPI::Wtime();
    request = MPI::COMM_WORLD.Isend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return {start, request};
}

std::pair<double, MPI::Request> ISsend(std::string& msg, int msg_tag)
{
    double start;
    MPI::Request request;

    start = MPI::Wtime();
    request = MPI::COMM_WORLD.Issend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return {start, request};
}

std::pair<double, MPI::Request> IBsend(std::string& msg, int msg_tag)
{
    double start;
    MPI::Request request;
    void * buf = new char[msg.length()];

    MPI::Attach_buffer(buf, msg.size() + MPI::BSEND_OVERHEAD);
    start = MPI::Wtime();
    request = MPI::COMM_WORLD.Ibsend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);
    MPI::Detach_buffer(buf);

    return {start, request};
}

std::pair<double, MPI::Request> IRsend(std::string& msg, int msg_tag)
{
    double start;
    MPI::Request request;

    start = MPI::Wtime();
    request = MPI::COMM_WORLD.Irsend(msg.c_str(), msg.length(), MPI::CHAR, ProcessCodes::RECEIVER, msg_tag);

    return {start, request};
}
