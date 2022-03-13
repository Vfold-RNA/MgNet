#!/bin/bash

lib_path=./bio/build/lib/libtool.a
std_version=-std=c++17
include_path=./bio/build/include

for DB_res_factor in 50
do
for kmean_cluster_size in 380
do
#DB_res_factor=50
#kmean_cluster_size=400
# finally decide using 50_380 combination, and use epoch 40
save_folder="./${DB_res_factor}_${kmean_cluster_size}/"
echo $save_folder
for cv_index in {1..5}
do
    #for folder_name in {20,40}
    #do

        #cubic_step=0.5

        #g++ density.cpp -o density ${std_version} ${lib_path} -I ${include_path} -lpthread -lstdc++fs

        #./density ./cv${cv_index}/${folder_name}/raw/ ${cubic_step}

        #python cluster.py input_folder output_folder DB_cluster_res_factor kmean_cluster_size
        #cv_index=1
        folder_name=40
        python cluster.py ./cv${cv_index}/${folder_name}/density/ ./cv${cv_index}/${folder_name}/cluster/ ${DB_res_factor} ${kmean_cluster_size}
        #mv ./cv${cv_index}/${folder_name}/cluster/ ./cv${cv_index}/${folder_name}/cluster.bak
    #done
done
cd ../analysis/; bash run.sh; mkdir ./${save_folder} ; cp cv_* ./${save_folder}/ ;
cd ../test_result/;

done
done
