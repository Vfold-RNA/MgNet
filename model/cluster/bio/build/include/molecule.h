#pragma once

#include "block.h"
#include "atom.h"

// #include <vector>
#include <string>
#include <map>
#include <set>
// #include <memory>
// #include <iostream>
// #include <algorithm>
// #include <cstring>//memcmp



namespace bio{

    typedef std::reference_wrapper<bio::Atom> ref_Atom;
    typedef Block<ref_Atom> Selection;

    //Molecule class
    class Molecule : public Block<Atom>{
        friend void ReadPDB(std::istream& in, Molecule& m);
        friend void ReadMol2(std::istream& in, Molecule& m);
    public:
        Molecule(){}
        Molecule(const std::string id) : id_(id){}
        Molecule(const std::string id, const std::string type) : id_(id), type_(type){}
        Molecule(std::initializer_list<Atom> il) : Block<Atom>(il){}
        Molecule(const Molecule& m) : Block<Atom>(m), id_(m.id_), type_(m.type_){}
        Molecule(const Selection& sel) {
            for(const auto& s : sel)
            this->push_back(s);
        };

        inline void set_id(const std::string id){id_ = id;}
        inline void set_type(const std::string type){type_ = type;}
        inline std::string get_id() const{return id_;};
        inline std::string get_type() const{return type_;};

        inline unsigned int get_model_num() const{
            std::set<unsigned int> model_set;
            for(const auto& a : *this){
                model_set.insert(a.model_serial);
            }
            return model_set.size();
        };
        inline unsigned int get_atom_num() const{return this->size();}
        
    private:
        
        std::string id_ = "";
        std::string type_ = "";

        // void preProcess();
    };

    //the powerful selection part is here
    Selection AtomSelect(Molecule& m, const std::string str);

        //SelectionParser class
    class SelectionParser{
        friend Selection AtomSelect(Molecule& m, const std::string str);
        public:
            SelectionParser(const std::string str);
        private:
            bool parse(const Atom& atom);
            bool parse_flag_ = false;
            std::map<std::string,std::string> sel_map_{{"name","none"},{"res_serial","none"},{"res_name","none"},{"chain_name","none"},{"serial","none"},{"model_serial","none"},{"alt_loc","none"},{"element","none"}};
    };

    // template<typename T>
    // class Molecule : public Block<T>{
    // public:
    //     Molecule(){}
    //     Molecule(char const* id) : Id(id){}
    //     Molecule(std::initializer_list<T> il) : Block<T>(il){}
    //     Molecule(const Molecule& m) : Block<T>(m){}

    //     inline void set_Id(char const* id){Id = std::string(id);}
        
    //     //the powerful selection part is here
    //     template<typename SEL>
    //     Molecule<SEL> select(const std::string selStr){
    //         selectionParser selParser(selStr);
    //         // typedef std::reference_wrapper<bio::Atom> selType;
    //         bio::Molecule<SEL> sel;
    //         for(auto a=0; a<(*this).size(); ++a){
    //             if(selParser.parse((*this)[a])){
    //                 //use [] operator will return a reference
    //                 //reference will be treat as value when it is inserted into the Selection<Atom>?
    //                 sel.push_back((*this)[a]);
    //             }
    //         }
    //         return sel;
    //     }
    // private:
    //     std::string Id;
    // };
}//namespace bio