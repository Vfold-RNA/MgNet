#!/bin/bash

std_version=-std=c++11
cd density_src ; g++ density.cpp create_directory.cpp file_list.cpp string2vector.cpp trim.cpp -o density ${std_version} -lpthread -lstdc++fs -O3 ; cd -
ln -s ./density_src/density ./density

for DB_res_factor in 50
do
for kmean_cluster_size in 380
do
save_folder="./${DB_res_factor}_${kmean_cluster_size}/"
echo $save_folder
for cv_index in {1..5}
do
    folder_name=40
    cubic_step=0.5
    ./density ./cv${cv_index}/${folder_name}/raw/ ${cubic_step}
    python cluster.py ./cv${cv_index}/${folder_name}/density/ ./cv${cv_index}/${folder_name}/cluster/ ${DB_res_factor} ${kmean_cluster_size} ${cv_index}

    rm -r ./cv${cv_index}/${folder_name}/raw/
    rm -r ./cv${cv_index}/${folder_name}/density/
done
#cd ../analysis/; bash run.sh; mkdir ./${save_folder} ; cp cv_* ./${save_folder}/ ;
#cd ../test_result/;
done
done

# average cluster
#cubic_step=0.5
#g++ average_density.cpp -o average_density ${std_version} ${lib_path} -I ${include_path} -lpthread -lstdc++fs -O3
#./average_density ${cubic_step}
#DB_res_factor=50
#kmean_cluster_size=380
#python average_cluster.py ${DB_res_factor} ${kmean_cluster_size}
