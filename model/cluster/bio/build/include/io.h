#pragma once

#include "atom.h"
#include "molecule.h"
#include "pdb.h"
#include "mol2.h"
#include "trim.h"

#include <string>
#include <iomanip>

#include <istream>
#include <ostream>
#include <fstream>

namespace bio{

    template <typename T>
	void ReadPDB(T& in, Molecule& m){
		if(!in){
			std::cout << " can not open for reading! Function ReadPDB" << std::endl;
			exit(2);
		}
		m.clear();
		std::string sline;
		int model_serial = 1;
		while(std::getline(in,sline)){
			switch (PDBParser::parseRECORD(sline)){
				case PDBParser::MODEL_:{
					model_serial = PDBParser::parseMODEL(sline);
					break;
				}
				case PDBParser::ENDMDL_:{
					model_serial++;
					break;
				}
				case PDBParser::ATOM_:{
					Atom a = PDBParser::parseATOM(sline);
					a.model_serial = model_serial;
					m.push_back(a);
					break;
				}
				case PDBParser::HETATM_:{
					Atom a = PDBParser::parseHETATM(sline);
					a.model_serial = model_serial;
					m.push_back(a);
					break;
				}
			}
		}
	}//ReadPDB

    template <typename T>
    void ReadMol2(T& in, Molecule& m){
        if(!in){
			std::cout << " can not open for reading! Function ReadMol2" << std::endl;
			exit(2);
		}
		m.clear();
		int model_serial = 0;
		std::string sline;
		MOL2Parser::BLOCK_TYPE BLOCKSTATE = MOL2Parser::UNKNOWN_;
        while(std::getline(in,sline)){
			if(util::Trim(sline).empty()) continue;
			MOL2Parser::BLOCK_TYPE PARSESTATE = MOL2Parser::parseTYPE(sline);
			BLOCKSTATE = (PARSESTATE==MOL2Parser::NONE_) ? BLOCKSTATE : PARSESTATE;
			if(PARSESTATE!=MOL2Parser::NONE_) std::getline(in,sline);
			switch (BLOCKSTATE){
				case MOL2Parser::MOLECULE_:{
					m.set_id(util::Trim(sline));
					std::getline(in,sline);
					// std::vector<std::string> V_num = String_to_Vector_String(sline);
					// t_model.TRIPOS_MOLECULE.num_atoms = std::stoi(V_num.at(0));
					// t_model.TRIPOS_MOLECULE.num_bonds = std::stoi(V_num.at(1));
					// t_model.TRIPOS_MOLECULE.num_subst = std::stoi(V_num.at(2));
					// t_model.TRIPOS_MOLECULE.num_feat = std::stoi(V_num.at(3));
					// t_model.TRIPOS_MOLECULE.num_sets = std::stoi(V_num.at(4));

					std::getline(in,sline);
					m.set_type(util::Trim(sline));

					std::getline(in,sline);
					// t_model.TRIPOS_MOLECULE.charge_type = sline;
					++model_serial;
                    break;
				}
				case MOL2Parser::ATOM_:{
					sline = util::Trim(sline);
					if(!sline.empty()){
						Atom a = MOL2Parser::parseATOM(sline);
						a.model_serial = model_serial;
						m.push_back(a);
					}
					break;
				}
				case MOL2Parser::BOND_:{
					break;
				}
				case MOL2Parser::SUBSTRUCTURE_:{
					break;
				}
				case MOL2Parser::UNKNOWN_:{
					break;
				}
				default:{
					std::cout << "mol2 default please check read.cpp!" << std::endl;
					break;
				}
			}//switch
		}//while
    }//ReadMol2

    template <typename T>
    void WriteMol2Atom(T& out, const Atom& a){
		out << "  " << a.serial << " " << a.name << " " << std::fixed << std::setprecision(4) << a.x << " " << a.y << " " << a.z << " " << a.sybyl_atom_type << " " << a.res_serial << " " << a.res_name << " " << a.partial_charge;
		out << std::defaultfloat << std::setprecision(6);
	}//WriteMol2Atom

    template <typename T>
    void WritePDBAtom(T& out, const Atom& a){
		if( a.element.size() == 1 && a.name.size() != 4 ){
		    out << std::setfill(' ') << std::setw(6) << std::left << a.record_name << std::right << std::setw(5) << a.serial << " "
		    << std::setw(4) << std::left << " "+a.name << std::right << std::setw(1) << a.alt_loc
		    << std::setw(3) << std::right << a.res_name << std::right << " " << std::setw(1) << a.chain_name 
		    << std::setw(4) << a.res_serial << std::setw(1) << a.i_code << std::setw(3) << " "
		    << std::setw(8) << std::fixed << std::setprecision(3) << a.x << std::setw(8) << a.y 
		    << std::setw(8) << a.z << std::setw(6) << std::setprecision(2) << a.occupancy 
		    << std::setw(6) << a.temp_factor << std::setw(10) << " " << std::setw(2) << a.element 
		    << std::setw(2) << a.charge;
		}else{
			out << std::setfill(' ') << std::setw(6) << std::left << a.record_name << std::right << std::setw(5) << a.serial << " " 
			<< std::setw(4) << std::left << a.name << std::right << std::setw(1) << a.alt_loc
			<< std::setw(3) << std::right << a.res_name << std::right << " " << std::setw(1) << a.chain_name 
			<< std::setw(4) << a.res_serial << std::setw(1) << a.i_code << std::setw(3) << " "
			<< std::setw(8) << std::fixed << std::setprecision(3) << a.x << std::setw(8) << a.y 
			<< std::setw(8) << a.z << std::setw(6) << std::setprecision(2) << a.occupancy 
			<< std::setw(6) << a.temp_factor << std::setw(10) << " " << std::setw(2) << a.element 
			<< std::setw(2) << a.charge;
		}
		out << std::defaultfloat << std::setprecision(6);
	}//WritePDBAtom

    //operator
    template <typename T>
	std::ostream& operator<<(T& out, const Atom& a){
		WritePDBAtom(out,a);
		return out;
	}

    template <typename T>
	std::istream& operator>>(T& in, Molecule& m){
		ReadPDB(in,m);
		return in;
	}//operator
}