#include <iostream>
#include <cmath>
#include <thread>
#include <string>
#include <vector>
#include <unordered_map>


extern "C" {

inline int Upper_Cubic_Boundary(const double coordinate, const double radius, const double step){
    return static_cast<int>(ceil((coordinate+radius)/step));
}
inline int Lower_Cubic_Boundary(const double coordinate, const double radius, const double step){
    return static_cast<int>(floor((coordinate-radius)/step));
}

void generate_atom_map(std::vector<std::unordered_map<std::string,double>> &atom_maps, const double cubic_step, float *coords, double *channelsigmas, int number_atoms, int number_channels){
    atom_maps = std::vector<std::unordered_map<std::string,double>>(number_channels,std::unordered_map<std::string,double>());
    for(int j = 0; j < number_atoms; j++){
        int atom_base = j*3;
        const double& atom_x = coords[atom_base+0];
        const double& atom_y = coords[atom_base+1];
        const double& atom_z = coords[atom_base+2];
        for(int k = 0; k < number_channels; k++){
            std::unordered_map<std::string,double>& atom_map = atom_maps[k];
            const double& vdw_radius = channelsigmas[j*number_channels+k];
            const double atom_radii = vdw_radius;
            //const double sigma = atom_radii/2;
            //const double gaussian_const_1d = 1/(sigma*std::sqrt(2*M_PI));
            //const double gaussian_const = gaussian_const_1d*gaussian_const_1d*gaussian_const_1d;

            const int lower_x = Lower_Cubic_Boundary( atom_x, atom_radii, cubic_step );
            const int lower_y = Lower_Cubic_Boundary( atom_y, atom_radii, cubic_step );
            const int lower_z = Lower_Cubic_Boundary( atom_z, atom_radii, cubic_step );
        
            const int upper_x = Upper_Cubic_Boundary( atom_x, atom_radii, cubic_step );
            const int upper_y = Upper_Cubic_Boundary( atom_y, atom_radii, cubic_step );
            const int upper_z = Upper_Cubic_Boundary( atom_z, atom_radii, cubic_step );

            for( int x = lower_x; x != upper_x; ++x )
            {
                for( int y = lower_y; y != upper_y; ++y )
                {
                    for( int z = lower_z; z != upper_z; ++z )
                    {
                        const double grid_center_x = (static_cast<double>(x)+0.5)*cubic_step;
                        const double grid_center_y = (static_cast<double>(y)+0.5)*cubic_step;
                        const double grid_center_z = (static_cast<double>(z)+0.5)*cubic_step;
                        const double r = std::sqrt((atom_x-grid_center_x)*(atom_x-grid_center_x) + (atom_y-grid_center_y)*(atom_y-grid_center_y) + (atom_z-grid_center_z)*(atom_z-grid_center_z));
                        if( r <= 1.5*atom_radii )
                        {
                            int count = 0;
                            for( int i = 0; i <= 1; i++ )
                            {
                                for( int j = 0; j <= 1; j++ )
                                {
                                    for( int k = 0; k <= 1; k++ )
                                    {
                                        const double smaller_grid_atom_x = (grid_center_x-0.25*cubic_step) + static_cast<double>(i)*0.5*cubic_step;
                                        const double smaller_grid_atom_y = (grid_center_y-0.25*cubic_step) + static_cast<double>(j)*0.5*cubic_step;
                                        const double smaller_grid_atom_z = (grid_center_z-0.25*cubic_step) + static_cast<double>(k)*0.5*cubic_step;
                                        const double dis = std::sqrt((atom_x-smaller_grid_atom_x)*(atom_x-smaller_grid_atom_x) + (atom_y-smaller_grid_atom_y)*(atom_y-smaller_grid_atom_y) + (atom_z-smaller_grid_atom_z)*(atom_z-smaller_grid_atom_z));
                                        if( dis <= 1.5*atom_radii )
                                        {
                                            count++;
                                        }
                                    }//End for k
                                }//End for j
                            }//End for i
                            if( count >= 4 )
                            {
                                std::string cell_name = std::to_string(x)+" "+std::to_string(y)+" "+std::to_string(z);
                                //3D gaussian distribution
                                //const double frac = gaussian_const*std::exp(-(r*r)/(2*sigma*sigma));
                                //kDeep distribution
                                const double frac = 1.0 - std::exp(-std::pow((atom_radii/r),12));
                                const auto cell_it = atom_map.find(cell_name);
                                if (cell_it == atom_map.end()){
                                    atom_map.insert(std::make_pair(cell_name,frac));
                                }else{
                                    if(cell_it->second<=frac){
                                        cell_it->second = frac;
                                    }
                                }
                            }
                        }//if r
                    }//z
                }//y
            }//x 

        }//k
    }//j
    
}



void descriptor_ext_thread(std::vector<std::unordered_map<std::string,double>> &atom_maps, const double cubic_step, double *centers, double *occus, int number_channels, int number_center_start, int number_center_end){
    for(int k = 0; k < number_channels; k++){
        const std::unordered_map<std::string,double>& atom_map = atom_maps[k];
        for(int i = number_center_start; i < number_center_end; i++){
            const int center_base = i*3;
            const int center_x = int(centers[center_base+0]/cubic_step);
            const int center_y = int(centers[center_base+1]/cubic_step);
            const int center_z = int(centers[center_base+2]/cubic_step);
            
            std::string center_cell = std::to_string(center_x)+" "+std::to_string(center_y)+" "+std::to_string(center_z);
            
            const auto it = atom_map.find(center_cell);
            if(it!=atom_map.end()){
                occus[i*number_channels+k] = it->second;
            }else{
                occus[i*number_channels+k] = 0;
            }
        }//i
    }//k
}//descriptor_ext_thread

void descriptor_ext(double *centers, float *coords, double *channelsigmas, double *occus, int number_centers, int number_atoms, int number_channels, double resolution, int thread_number){
    std::vector<std::unordered_map<std::string,double>> atom_maps;
    generate_atom_map(atom_maps, resolution, coords, channelsigmas, number_atoms, number_channels);

    std::thread t[thread_number];
    int thread_size = int(number_centers/thread_number);
    for(int i = 0; i != thread_number; ++i){
        int number_center_start = thread_size*i;
        int number_center_end = (i==(thread_number-1)) ? number_centers : thread_size*(i+1);
        t[i] = std::thread( descriptor_ext_thread, std::ref(atom_maps), resolution, centers, occus, number_channels, number_center_start, number_center_end );
    }
    for(int i = 0; i != thread_number; ++i){
        t[i].join();
    }
}

}