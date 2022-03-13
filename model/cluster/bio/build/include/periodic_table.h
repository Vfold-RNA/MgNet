#pragma once

//#include <vector>
//#include <string>
// #include <memory>
// #include <iostream>
// #include <algorithm>
//#include <cmath>//M_PI
//#include <map>
//#include <set>

namespace bio{
/*
 * periodic table of elements and helper functions to convert
 * ordinal numbers to labels and back.
 *
 */

const char *get_pte_label(const int idx);
float get_pte_mass(const int idx);
float get_pte_vdw_radius(const int idx);
int get_pte_idx_from_string(const char *label);
}//namespace
