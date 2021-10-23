#!/bin/bash
srun \
 --time=00:40:00\
 --container-mounts=/scratch:/scratch,/home/kvedder/code/bag_processing:/project,/Datasets:/Datasets\
 --container-image=ros:melodic-ros-base-bionic \
bash -c "cd /project; ./dump_bag_data.py2 /scratch/kvedder/ros_raw_data/ ./dump_bag_data.py2 data/data_dump_raw/ /scratch/kvedder/_2021-10-15*.bag"
