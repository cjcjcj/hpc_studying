#pragma once

#include "mpi.h"

#include <string>
#include <functional>
#include <utility>

using send_function_t = std::function<double(std::string&, int)>;
using isend_function_t = std::function<std::pair<double, MPI::Request>(std::string&, int)>;


double Send(std::string& msg, int msg_tag);
double Ssend(std::string& msg, int msg_tag);
double Bsend(std::string& msg, int msg_tag);
double Rsend(std::string& msg, int msg_tag);

std::pair<double, MPI::Request> ISend(std::string& msg, int msg_tag);
std::pair<double, MPI::Request> ISsend(std::string& msg, int msg_tag);
std::pair<double, MPI::Request> IBsend(std::string& msg, int msg_tag);
std::pair<double, MPI::Request> IRsend(std::string& msg, int msg_tag);
