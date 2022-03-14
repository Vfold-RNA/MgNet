#include "string2vector.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
namespace util{
   std::vector<std::string> String2Vector(const std::string sline, int outlet){
      std::istringstream ss(sline);
      std::string buf;
      std::vector<std::string> token;
      while(ss >> buf) token.push_back(buf);
      if(outlet == 1){
         for( const auto &tok : token )
            std::cout << tok << " ";

         std::cout << std::endl;
      }
      return token;
   }
}//namespace
