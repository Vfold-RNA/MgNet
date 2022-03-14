#include "string2vector.h"
#include "trim.h"
#include "file_list.h"
#include "create_directory.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <unordered_map>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int main(int argc, char** argv){
    std::cout << "Have " << argc << " arguments:" << std::endl;
    for (int i = 0; i < argc; ++i){
        std::cout << argv[i] << std::endl;
    }
    const double cubic_step = std::stod(argv[1]);
    const std::string fileExt = ".csv";

    std::map<std::string,std::vector<std::string>> m_prediction_file_list;
    for(int i = 1; i <= 5; ++i) {
        const std::string raw_file_dir = "./cv"+std::to_string(i)+"/40/raw/";
        util::File_List_Args args; args.dir=raw_file_dir; args.mode="shallow"; args.extension=fileExt; args.flag=0;
        std::vector<util::struct_file> prediction_file_list = util::FileList(args);
        for(const auto& file : prediction_file_list) {
            std::cout << file.name << std::endl;
            if(m_prediction_file_list.find(file.name) != m_prediction_file_list.end()) {
                m_prediction_file_list[file.name].push_back(file.dir+file.name);
            } else {
                m_prediction_file_list.insert({file.name, {file.dir+file.name}});
            }
        }
    }

    for(const auto& name_file_list_pair : m_prediction_file_list) {
        const std::string& file_name = name_file_list_pair.first;
        std::cout << file_name << " " << name_file_list_pair.second.size() << std::endl;

        std::map<std::string,std::pair<int,double>> UM_site;
        for(const std::string& file_path : name_file_list_pair.second) {
            std::ifstream in(file_path);
            std::string line;
            while(std::getline(in,line)){
                if(line.size()!=0){
                    std::vector<std::string> vs = util::String2Vector(line);
                    std::string site = vs[0]+" "+vs[1]+" "+vs[2];
                    const double score = std::stod(vs[3]);
                    const auto it = UM_site.find(site);
                    if(it!=UM_site.end()){
                        it->second.first++;
                        it->second.second+=score;
                    }else{
                        UM_site.insert({site,{1,score}});
                    }
                }
            }
            in.close();
            std::cout << "UM_site_size " << UM_site.size() << std::endl;
        }

        std::cout << "reorder..." << std::endl;
        std::multimap<double,std::string> M_site;
        for(const auto &s : UM_site){
            M_site.insert({-s.second.second/double(s.second.first),s.first});
        }
        
        util::Create_Directory("./average_result/");
        std::ofstream xyz_out("./average_result/"+file_name.substr(0,file_name.find(fileExt))+".xyz");
        int count1 = 0;
        for(const auto &s : M_site){
            count1++;
            std::vector<std::string> vs = util::String2Vector(s.second);
            
            const double x = (std::stod(vs[0]))*cubic_step;
            const double y = (std::stod(vs[1]))*cubic_step;
            const double z = (std::stod(vs[2]))*cubic_step;
            const double score = -s.first;
            xyz_out << x << " " << y << " " << z << " " << score << std::endl;
        }
        xyz_out.close();
        // std::cout << "output..." << std::endl;
        // std::ofstream out(file.dir+"../prediction/"+std::to_string(num_hits_each_RES)+"/"+file.name.substr(0,file.name.find(fileExt))+".pdb");
        // int count = 0;
        // for(const auto &s : M_site){
        //     std::vector<std::string> vs = util::String2Vector(s.second);
        //     count++;
        //     pdb::Atom a;
        //     a.name = "MG";
        //     a.element = "Mg";
        //     a.serial = count % 1000;
        //     a.resName = "MG";
        //     a.resSeq = count % 1000;
        //     a.chainID = "Z";
        //     a.recordName = "HETATM";
        //     a.occupancy = 1.00;
        //     // a.charge = "2.0";
        //     a.x = (std::stod(vs[0]))*cubic_step;
        //     a.y = (std::stod(vs[1]))*cubic_step;
        //     a.z = (std::stod(vs[2]))*cubic_step;
        //     a.tempFactor = -s.first;
        //     out << a;
        // }
        // out.close();
    }

    return 0;
}
