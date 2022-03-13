#pragma once

#include "molecule.h"
#include "atom.h"
// #include "io.h"

// #include <iostream>
// #include <fstream>
#include <string>
// #include <vector>

#include <regex>

namespace bio{
// class PDB{
// friend std::ifstream& operator>>(std::ifstream&, PDB&);
// public:
//     static int PDB_number;

//     PDB(){}
//     PDB(const std::string path){ std::ifstream inPDB(path,std::ios::in); inPDB >> *this; }
//     //-------------------------------------------------------------------
// private:
//     std::vector<std::string> content;
// protected:
// };
// std::ifstream& operator>>(std::ifstream&, PDB&);
// std::ostream& operator<<(std::ostream& str, const PDB& p);
    
	class PDBParser{
		// friend std::istream& operator>>(std::istream& in, Molecule& m);
		template <typename T>
		friend void ReadPDB(T& in, Molecule& m);
	private:
		enum RECORD_TYPE{CRYST1_,END_,HEADER_,NUMMDL_,MASTER_,ORIGXn_,SCALEn_,AUTHOR_,CAVEAT_,COMPND_,EXPDTA_,MDLTYP_,KEYWDS_,OBSLTE_,SOURCE_,SPLIT_,SPRSDE_,TITLE_,ANISOU_,ATOM_,HETATM_,CISPEP_,CONECT_,DBREF_,HELIX_,HET_,LINK_,MODRES_,MTRIXn_,REVDAT_,SEQADV_,SHEET_,SSBOND_,FORMUL_,HETNAM_,HETSYN_,SEQRES_,SITE_,ENDMDL_,MODEL_,TER_,JRNL_,REMARK_};

		const static std::regex r_CRYST1_;
		const static std::regex r_END_;
		const static std::regex r_HEADER_;
		const static std::regex r_NUMMDL_;
		const static std::regex r_MASTER_;
		const static std::regex r_ORIGXn_;
		const static std::regex r_SCALEn_;
		const static std::regex r_AUTHOR_;
		const static std::regex r_CAVEAT_;
		const static std::regex r_COMPND_;
		const static std::regex r_EXPDTA_;
		const static std::regex r_MDLTYP_;
		const static std::regex r_KEYWDS_;
		const static std::regex r_OBSLTE_;
		const static std::regex r_SOURCE_;
		const static std::regex r_SPLIT_;
		const static std::regex r_SPRSDE_;
		const static std::regex r_TITLE_;
		const static std::regex r_ANISOU_;
		const static std::regex r_ATOM_;
		const static std::regex r_HETATM_;
		const static std::regex r_CISPEP_;
		const static std::regex r_CONECT_;
		const static std::regex r_DBREF_;
		const static std::regex r_HELIX_;
		const static std::regex r_HET_;
		const static std::regex r_LINK_;
		const static std::regex r_MODRES_;
		const static std::regex r_MTRIXn_;
		const static std::regex r_REVDAT_;
		const static std::regex r_SEQADV_;
		const static std::regex r_SHEET_;
		const static std::regex r_SSBOND_;
		const static std::regex r_FORMUL_;
		const static std::regex r_HETNAM_;
		const static std::regex r_HETSYN_;
		const static std::regex r_SEQRES_;
		const static std::regex r_SITE_;
		const static std::regex r_ENDMDL_;
		const static std::regex r_MODEL_;
		const static std::regex r_TER_;
		const static std::regex r_JRNL_;
		const static std::regex r_REMARK_;

		const RECORD_TYPE static parseRECORD(const std::string &line);
		const int static parseMODEL(const std::string &line);
		static void parseENDMDL(const std::string &line);
		const Atom static parseATOM(const std::string &line);
		const Atom static parseHETATM(const std::string &line);
	};

}//namespace bio