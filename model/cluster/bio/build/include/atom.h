#pragma once

#include <vector>
#include <string>
// #include <memory>
// #include <iostream>
// #include <algorithm>
#include <cmath>//M_PI
#include <map>
#include <set>

namespace bio{

    enum ATOM_TYPE {H_,HE_,LI_,BE_,B_,C_,N_,O_,F_,NE_,NA_,MG_,AL_,SI_,P_,S_,CL_,AR_,K_,CA_,SC_,TI_,V_,CR_,MN_,FE_,CO_,NI_,CU_,ZN_,GA_,GE_,AS_,SE_,BR_,KR_,RB_,SR_,Y_,ZR_,NB_,MO_,TC_,RU_,RH_,PD_,AG_,CD_,IN_,SN_,SB_,TE_,I_,XE_,CS_,BA_,LA_,CE_,PR_,ND_,PM_,SM_,EU_,GD_,TB_,DY_,HO_,ER_,TM_,YB_,LU_,HF_,TA_,W_,RE_,OS_,IR_,PT_,AU_,HG_,TL_,PB_,BI_,PO_,AT_,RN_,FR_,RA_,AC_,TH_,PA_,U_,NP_,PU_,AM_,CM_,BK_,CF_,ES_,FM_,MD_,NO_,LR_,RF_,DB_,SG_,BH_,HS_,MT_,DS_,RG_,CN_};

    class Atom{
    // friend class Radius;
    friend class Metal;
    // enum ATOM_TYPE {H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn};

    public:
        std::string record_name = "";
        int serial;
        std::string name = "";
        std::string alt_loc = "";
        std::string res_name = "";
        std::string chain_name = "";
        int res_serial;
        std::string i_code = "";
        double x;
        double y;
        double z;
        double occupancy = 0.0;
        double temp_factor = 0.0;
        std::string element = "";
        std::string charge = "";

        double partial_charge = 0.0;
        std::string sybyl_atom_type = "";

        unsigned int model_serial = 1;

        // Atom(std::initializer_list<std::string>);
        // Atom(const Atom&);
        const double get_covalent_radius() const;
        const double get_atomic_radius() const;
        const double get_vdw_radius() const;
        const bool is_metal() const;
    private:
    };

    // bool operator==(const Atom&, const Atom&);
    // bool operator!=(const Atom&, const Atom&);
    inline const double Distance(const Atom &a, const Atom &b){
        return std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
    }//End PDB::distance
    // inline const double angle(const Atom&, const Atom&, const Atom&, const std::string mode = "arc", const double PI = M_PI );

    class Radius{
        friend class Atom;
    public:
        Radius();
    private:
        std::map<std::string,double> covalent_;
        std::map<std::string,double> atomic_;
        std::map<std::string,double> vdw_;
    };

    extern Radius radii_;

    class Metal{
        friend class Atom;
        private:
            // enum METAL_TYPE {AC_,AG_,AL_,AM_,AU_,BA_,BE_,BI_,BK_,CA_,CD_,CE_,CF_,CM_,CO_,CR_,CS_,CU_,DB_,DY_,ER_,ES_,EU_,FE_,FM_,FR_,GA_,GD_,GE_,HF_,HG_,HO_,IN_,IR_,K_,LA_,LI_,LR_,LU_,MD_,MG_,MN_,MO_,NA_,NB_,ND_,NI_,NO_,NP_,OS_,PA_,PB_,PD_,PM_,PO_,PR_,PT_,PU_,RA_,RB_,RE_,RF_,RH_,RU_,SB_,SC_,SG_,SM_,SN_,SR_,TA_,TB_,TC_,TH_,TI_,TL_,TM_,U_,V_,W_,Y_,YB_,ZN_,ZR_};
            std::set<std::string> table_{"AC","AG","AL","AM","AU","BA","BE","BI","BK","CA","CD","CE","CF","CM","CO","CR","CS","CU","DB","DY","ER","ES","EU","FE","FM","FR","GA","GD","GE","HF","HG","HO","IN","IR","K","LA","LI","LR","LU","MD","MG","MN","MO","NA","NB","ND","NI","NO","NP","OS","PA","PB","PD","PM","PO","PR","PT","PU","RA","RB","RE","RF","RH","RU","SB","SC","SG","SM","SN","SR","TA","TB","TC","TH","TI","TL","TM","U","V","W","Y","YB","ZN","ZR"};
        public:
            // static std::vector<METAL_TYPE> metal_list();
    };

    extern Metal metal_;
}