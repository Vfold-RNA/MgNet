#pragma once

#include "molecule.h"
#include "atom.h"
// #include "io.h"

// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <map>

// #include <iomanip>

#include <regex>

namespace bio{
    class MOL2Parser{
		template <typename T>
        friend void ReadMol2(T& in, Molecule& m);
	private:
		enum BLOCK_TYPE{MOLECULE_,ATOM_,BOND_,SUBSTRUCTURE_,NONE_,UNKNOWN_};

		const static std::regex r_MOLECULE_;
		const static std::regex r_ATOM_;
        const static std::regex r_BOND_;
        const static std::regex r_SUBSTRUCTURE_;
        const static std::regex r_UNKNOWN_;

		const BLOCK_TYPE static parseTYPE(const std::string &line);
		// const int static parseMODEL(const std::string &line);
		// static void parseENDMDL(const std::string &line);
        static void parseMOLECULE(const std::string &sline);
		const Atom static parseATOM(const std::string &line);
        // const Bond static parseBond(const std::string &line);
	};
}//namespace bio