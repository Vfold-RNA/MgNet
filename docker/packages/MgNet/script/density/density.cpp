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
    const std::string raw_file_dir = argv[1];
    const double cubic_step = std::stod(argv[2]);

    const std::string fileExt = ".csv";

    util::File_List_Args args; args.dir=raw_file_dir; args.mode="shallow"; args.extension=fileExt; args.flag=0;
    std::vector<util::struct_file> prediction_file_list = util::FileList(args);

    for(const auto &file : prediction_file_list){
        std::cout << file.name << std::endl;
        // pdb::PDB expt("/home/yuanzhe/research/metalIon/data/original/",file.name.substr(0,4)+".pdb");
        // expt.get_polymer_structure();
        // std::vector<pdb::Model> expt_molecule_models = expt.polymer_structure.get_list();
        // pdb::Model& expt_molecule_model = expt_molecule_models[0];
        // unsigned int residue_num = 0;
        // for(const auto &chain : expt_molecule_model.get_list()){
        //     for(const auto &residue : chain.get_list()){
        //         if(residue.is_nucleotide()){
        //             residue_num++;
        //         }
        //     }
        // }
		// std::vector<pdb::Atom> exptAtoms = expt_molecule_model.find_atom_by_name("MG");

        std::map<std::string,std::pair<int,double>> UM_site;
        std::ifstream in(file.dir+file.name);
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

        std::cout << "reorder..." << std::endl;
        std::multimap<double,std::string> M_site;
        for(const auto &s : UM_site){
            M_site.insert({-s.second.second/double(s.second.first),s.first});
        }

        util::Create_Directory(file.dir+"../density/");
        std::ofstream xyz_out(file.dir+"../density/"+file.name.substr(0,file.name.find(fileExt))+".xyz");
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
