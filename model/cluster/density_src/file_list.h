#pragma once

#include <vector>
#include <string>
#include <iostream>
namespace util{
    // template <typename T>
    struct File_List_Args
    {
        std::string extension = "all";
        std::string mode = "shallow";
        std::string dir = "./";
        bool flag = false;
        std::ostream* output = &std::cout;
    };
    // template struct File_List_Args<std::ostream>; 
    // template struct File_List_Args<std::ofstream>;

    struct struct_file
    {
        std::string dir;
        // std::string path;
        std::string name;
        // int number;
        // static int file_total_number;
    };

    std::vector<struct_file> FileList ( const File_List_Args& Args = File_List_Args() );
}//namespace