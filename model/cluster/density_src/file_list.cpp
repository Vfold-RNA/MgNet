#include "file_list.h"

#include <experimental/filesystem>

#include <iostream>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <algorithm>



namespace util{
    // int struct_file::file_total_number = 0;
    //return files in current directory
    std::vector<struct_file> Shallow_File_List ( std::string dir, std::string extension )
    {
        struct_file temp_file = {};
        std::vector<struct_file> files;
        DIR *dp = NULL;
        DIR *dp_tmp = NULL;
        struct dirent *dirp = NULL;
        if((dp  = opendir(dir.c_str())) == NULL) 
        {
            std::cout << "Error(" << errno << ") opening " << dir << std::endl;
            //return errno;
        }

        while ( (dirp = readdir(dp)) != NULL ) 
        {
            std::string dir_tmp = std::string(dirp->d_name);  //std::cout << dir_tmp << std::endl;
            if(dir_tmp != "." && dir_tmp != "..")
            {
                if((dp_tmp  = opendir(dir_tmp.c_str())) == NULL)
                {
                    temp_file = {};
                    temp_file.dir = dir;
                    temp_file.name = dir_tmp;
                    // temp_file.path = dir + dir_tmp;
                    // temp_file.number = ++(struct_file::file_total_number);
                    files.push_back(temp_file);
                }
                else
                {closedir(dp_tmp);}
            }
        }
        closedir(dp);

        if( extension != "all" && extension != "ALL" )
        {
            //Remove all files that are not with extension
            for(std::vector<struct_file>::iterator it = files.begin(); it != files.end(); ) 
            {
                struct_file file_examine = *it;
                std::string file_name = file_examine.name;
                //If this is not extension, we don't need
                if( file_name.find(extension.c_str()) == std::string::npos || file_name.find("~") != std::string::npos )
                {
                    it = files.erase(it);
                }else
                {
                    it++;
                }
            }
        }else
        {
            //collect all files
            for(std::vector<struct_file>::iterator it = files.begin(); it != files.end(); ) 
            {
                struct_file file_examine = *it;
                std::string file_name = file_examine.name;

                std::experimental::filesystem::path root_dir_path( dir+file_name );
                if( file_name.find("~") != std::string::npos || std::experimental::filesystem::is_directory(root_dir_path) )
                {
                    it = files.erase(it);
                }else
                {
                    it++;
                }
            }
        }

        std::sort(files.begin(), files.end(),[](const struct struct_file &a, const struct struct_file &b){return ((a.dir+a.name) < (b.dir+b.name));});

        return files;
    }


    //return files in current directory and all subdirectories
    std::vector<struct_file> Deep_File_List ( std::string dir, std::string extension )
    {
        std::vector<struct_file> files; files.clear();

        std::vector<struct_file> t_files = Shallow_File_List(dir,extension);
        files.insert(std::end(files), std::begin(t_files), std::end(t_files));
        

        std::experimental::filesystem::path root_dir_path( dir );
        std::experimental::filesystem::recursive_directory_iterator root_dir_iterator(root_dir_path);
        
        for(auto& d: root_dir_iterator)
        {
            std::experimental::filesystem::path temp_path = d.path();
            if( std::experimental::filesystem::is_directory(temp_path) )
            {
                std::string temp_dir = temp_path.string() + "/";
                
                std::vector<struct_file> t_files = Shallow_File_List(temp_dir,extension);
                files.insert(std::end(files), std::begin(t_files), std::end(t_files));
            }
        }

        return files;
    }


    std::vector<struct_file> FileList ( const File_List_Args& Args )
    {
        enum MODE {DEEP,SHALLOW};
        MODE TEMP_MODE = (Args.mode == "deep") ? DEEP : SHALLOW;
        switch( TEMP_MODE )
        {
            case DEEP:
            {
                std::vector<struct_file> files = Deep_File_List(Args.dir,Args.extension);
                if( Args.flag != false )
                {
                    for( auto i = 0; i != files.size(); ++i )
                    {
                        *(Args.output) << i+1 << " " << files[i].name << " " << files[i].dir << std::endl;
                    }
                }
                return files;
            }
                break;
            case SHALLOW:
            {
                std::vector<struct_file> files = Shallow_File_List(Args.dir,Args.extension);
                if( Args.flag != false )
                {
                    for( auto i = 0; i != files.size(); ++i )
                    {
                        *(Args.output) << i+1 << " " << files[i].name << " " << files[i].dir << std::endl;
                    }
                }
                return files;
            }
                break;
            default:
                break;
        }
    }
}//namespace