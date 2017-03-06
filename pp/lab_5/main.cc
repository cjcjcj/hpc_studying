#include "sendfunctions.h"
#include "constants.h"
#include "utils.h"

#include "mpi.h"

#include <iostream>
#include <string>
#include <vector>

void mpi_blocking(std::vector<size_t>& lengths_powers, int iterations_num=5)
{
    int 
        msg_tag,
        rank = MPI::COMM_WORLD.Get_rank();
    double start, end, time;
    std::string msg;

    // only for sender
    std::map<SendTypes, send_function_t> send_functions = {
        {SendTypes::SEND, Send},
        {SendTypes::SSEND, Ssend},
        {SendTypes::BSEND, Bsend},
        {SendTypes::RSEND, Rsend},
    };
    
    // only for receiver
    char *buf;
    MPI::Status status;
    int msg_len;

    for(auto lp: lengths_powers)
    {
        // create message only for sender
        if(rank == ProcessCodes::SENDER)
            msg = random_string(1 << lp);
        
        for(auto const& sf_ent : send_functions) {
        for(int i = 0; i < iterations_num; i++)
        {
            msg_tag = lp + i + sf_ent.first;

            switch(rank)
            {
                case ProcessCodes::SENDER:
                    // time
                    time = (sf_ent.second)(msg, msg_tag);
                    break;
                case ProcessCodes::RECEIVER:
                    MPI::COMM_WORLD.Probe(ProcessCodes::SENDER, msg_tag, status);
                    msg_len = status.Get_count(MPI::CHAR);
                    buf = new char[msg_len];

                    start = MPI::Wtime();
                    MPI::COMM_WORLD.Recv(buf, msg_len, MPI::CHAR, ProcessCodes::SENDER, msg_tag, status);
                    end = MPI::Wtime();
                    // time
                    time = end - start;

                    msg = std::string(buf, msg_len);
                    delete[] buf;
                    break;
            }

            std::cout << "block,"
                      << sf_ent.first << ","
                      << (rank == ProcessCodes::SENDER ? "sended" : "received")
                      // in bytes
                      << "," << msg.size()
                      // time computes in each case
                      << "," << time << std::endl;
        }
        }
    }
}

void mpi_nonblocking(std::vector<size_t>& lengths_powers, int iterations_num=5)
{
    int 
        msg_tag,
        rank = MPI::COMM_WORLD.Get_rank();
    double start, time;
    std::string msg;

    MPI::Request request;

    // only for sender
    std::map<SendTypes, isend_function_t> isend_functions = {
        {SendTypes::ISEND, ISend},
        {SendTypes::ISSEND, ISsend},
        {SendTypes::IBSEND, IBsend},
        {SendTypes::IRSEND, IRsend},
    };
    
    // only for receiver
    MPI::Status status;
    char *buf;
    int msg_len;

    for(auto lp: lengths_powers)
    {
        // create message only for sender
        if(rank == ProcessCodes::SENDER)
            msg = random_string(1 << lp);
        
        for(auto const& sf_ent : isend_functions) {
        for(int i = 0; i < iterations_num; i++)
        {
            msg_tag = lp + i + sf_ent.first;

            switch(rank)
            {
                case ProcessCodes::SENDER:
                    // time
                    {
                    auto send_result = (sf_ent.second)(msg, msg_tag);
                    start = send_result.first;
                    request = send_result.second;
                    }

                    request.Wait();
                    time = MPI::Wtime() - start;
                    break;
                case ProcessCodes::RECEIVER:
                    MPI::COMM_WORLD.Probe(ProcessCodes::SENDER, msg_tag, status);
                    msg_len = status.Get_count(MPI::CHAR);
                    buf = new char[msg_len];

                    start = MPI::Wtime();
                    request = MPI::COMM_WORLD.Irecv(buf, msg_len, MPI::CHAR, ProcessCodes::SENDER, msg_tag);

                    request.Wait();
                    time = MPI::Wtime() - start;

                    msg = std::string(buf, msg_len);
                    delete[] buf;
                    break;
            }

            std::cout << "nonblock,"
                      << sf_ent.first << ","
                      << (rank == ProcessCodes::SENDER ? "sended" : "received")
                      // in bytes
                      << "," << msg.size()
                      << "," << time << std::endl;
        }
        }
    }
}

int main(int argc, char** argv)
{
    int size;
    int iterations_num = 5;
    std::vector<size_t> lengths_powers = {20, 22, 24};

    MPI::Init();

    size = MPI::COMM_WORLD.Get_size();    

    if (size != 2)
    {
        std::cout << "World size must be 2\n";
        MPI::COMM_WORLD.Abort(1);
    }
    mpi_blocking(lengths_powers, iterations_num);
    mpi_nonblocking(lengths_powers, iterations_num);

    MPI::Finalize();
    return 0;
}
