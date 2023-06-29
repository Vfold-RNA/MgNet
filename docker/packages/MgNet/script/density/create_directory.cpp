#include "create_directory.h"

#include <string>
#include <iostream>
#include <experimental/filesystem>//create new directory
namespace util{
    void Create_Directory(const std::string dir_path){
        std::experimental::filesystem::path filesystem_dir_path(dir_path);
        if(!(std::experimental::filesystem::exists(filesystem_dir_path))){
            std::cout << filesystem_dir_path << "--------Doesn't Exists!" << std::endl;
            if(std::experimental::filesystem::create_directories(filesystem_dir_path)){
                std::cout << filesystem_dir_path <<"......Successfully Created !" << std::endl;
            }   
        }
    }
}//namespace
