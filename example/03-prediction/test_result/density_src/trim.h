#pragma once

#include <string>

namespace util{
    std::string Trim(const std::string &str, const std::string &whitespace = " \t");

    std::string Left_Trim(const std::string &str, const std::string &whitespace = " \t");

    std::string Right_Trim(const std::string &str, const std::string &whitespace = " \t");

    std::string Reduce(const std::string& str, const std::string& fill = " ", const std::string& whitespace = " \t");
}//namespace