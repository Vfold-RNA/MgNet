#include "trim.h"

#include <iostream>
#include <string>
namespace util{
    std::string Trim(const std::string &str, const std::string &whitespace){
        const auto str_begin = str.find_first_not_of(whitespace);
        if (str_begin == std::string::npos){
            return ""; // no content
        }

        const auto str_end = str.find_last_not_of(whitespace);
        const auto str_range = str_end - str_begin + 1;

        return str.substr(str_begin, str_range);
    }

    std::string Left_Trim(const std::string &str, const std::string &whitespace){
        const auto str_begin = str.find_first_not_of(whitespace);
        if (str_begin == std::string::npos){
            return ""; // no content
        }

        const auto str_end = str.size();
        const auto str_range = str_end - str_begin;

        return str.substr(str_begin, str_range);
    }

    std::string Right_Trim(const std::string &str, const std::string &whitespace){
        std::string::size_type str_begin = 0;
        const auto str_end = str.find_last_not_of(whitespace);
        if (str_end == std::string::npos){
            return ""; // no content
        }
        const auto str_range = str_end - str_begin + 1;
        return str.substr(str_begin, str_range);
    }

    std::string Reduce(const std::string& str, const std::string& fill, const std::string& whitespace){
        // Trim first
        auto result = Trim(str, whitespace);

        // replace sub ranges
        auto beginSpace = result.find_first_of(whitespace);
        while (beginSpace != std::string::npos){
            const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
            const auto range = endSpace - beginSpace;

            result.replace(beginSpace, range, fill);

            const auto newStart = beginSpace + fill.length();
            beginSpace = result.find_first_of(whitespace, newStart);
        }
        return result;
    }
}//namespace