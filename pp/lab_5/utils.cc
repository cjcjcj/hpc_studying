#include "constants.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <random>
#include <algorithm>

std::ostream& operator<< (std::ostream& os, SendTypes type)
{
    switch (type)
    {
        case SendTypes::SEND : return os << "Send" ;
        case SendTypes::SSEND: return os << "SSend";
        case SendTypes::BSEND: return os << "BSend";
        case SendTypes::RSEND: return os << "RSend";

        case SendTypes::ISEND : return os << "ISend" ;
        case SendTypes::ISSEND: return os << "ISSend";
        case SendTypes::IBSEND: return os << "IBSend";
        case SendTypes::IRSEND: return os << "IRSend";
        // omit default case to trigger compiler warning for missing cases
    };
    return os << type;
}

std::string random_string(size_t length)
{
    auto randchar = []() -> char
    {
        const char charset[] = 
        "0123456789"
        "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        "abcdefghiklmnoprqstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[std::rand() % max_index];
    };

    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}
