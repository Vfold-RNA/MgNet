#include <iostream>
#include <cmath>
#include <thread>
#include <vector>


extern "C" {
void descriptor_ext_thread(double *centers, float *coords, double *channelsigmas, double *channelvalues, double *occus, int number_centers, int number_atoms, int number_channels, double max_vdw, int number_center_start, int number_center_end){
    for(int i = number_center_start; i < number_center_end; i++){
        const int center_base = i*3;
        const double& center_x = centers[center_base+0];
        const double& center_y = centers[center_base+1];
        const double& center_z = centers[center_base+2];

            std::vector<int> effective_atom_num(number_channels,0);
            for(int j = 0; j < number_atoms; j++){
                int atom_base = j*3;
                const double& atom_x = coords[atom_base+0];
                const double& atom_y = coords[atom_base+1];
                const double& atom_z = coords[atom_base+2];
                const double dis = std::sqrt((atom_x-center_x)*(atom_x-center_x) + (atom_y-center_y)*(atom_y-center_y) + (atom_z-center_z)*(atom_z-center_z));
                if(dis > 1.1*max_vdw){
                    continue;
                }
                
                for(int k = 0; k < number_channels; k++){               
                    const double& vdw_radius = channelsigmas[j*number_channels+k];
                    const double& values = channelvalues[j*number_channels+k];
                    if(dis > vdw_radius){
                        continue;
                    }
                    if(vdw_radius > 0){
                        const double occupancy = values * (1.0 - std::exp(-std::pow((vdw_radius/dis),12)));
                        
                        //const double sigma = vdw_radius/2.5;
                        //const double N = 1/(sigma*sigma*sigma*std::pow(2*M_PI,1.5));
                        //const double occupancy = 10*values * N * std::exp(-dis*dis/(2*sigma*sigma));
                        occus[i*number_channels+k] += occupancy;
                        effective_atom_num[k]++;
                        //if(occus[i*number_channels+k] < occupancy){
                        //    occus[i*number_channels+k] = occupancy;
                        //    effective_atom_num++;
                        //}
                    }
                }//k
            }//j
            for(int k = 0; k < number_channels; k++){
                if(effective_atom_num[k] > 1){
                    occus[i*number_channels+k] = occus[i*number_channels+k]/effective_atom_num[k];
                }
            }//k
    }//i
}//descriptor_ext_thread

void descriptor_ext(double *centers, float *coords, double *channelsigmas, double *channelvalues, double *occus, int number_centers, int number_atoms, int number_channels, int thread_number){
    double max_vdw = -1;
    for(int j = 0; j < number_atoms; j++){
        for(int k = 0; k < number_channels; k++){
            const double& vdw_radius = channelsigmas[j*number_channels+k];
            if(max_vdw < vdw_radius){
                max_vdw = vdw_radius;
            }
        }//k
    }//j

    std::thread t[thread_number];
    int thread_size = int(number_centers/thread_number);
    for(int i = 0; i != thread_number; ++i){
        int number_center_start = thread_size*i;
        int number_center_end = (i==(thread_number-1)) ? number_centers : thread_size*(i+1);
        t[i] = std::thread( descriptor_ext_thread, centers, coords, channelsigmas, channelvalues, occus, number_centers, number_atoms, number_channels, max_vdw, number_center_start, number_center_end );
    }
    for(int i = 0; i != thread_number; ++i){
        t[i].join();
    }
}


void test(double t[][2], int row, int column){
    for(int i = 0; i < row; i++){
        std::cout << t[i][0] << std::endl;
    }
}

}
