# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/refine_registration.py

import numpy as np
import open3d as o3d
import copy
import argparse


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def multiscale_icp(source,
                   target,
                   voxel_size=0.2,
                   max_iter=[50, 30, 14],
                   init_transformation=np.identity(4)):
    voxel_size = [voxel_size, voxel_size / 2.0, voxel_size / 4.0]
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = 0.5
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                 max_nn=60))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                 max_nn=60))
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, distance_threshold,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))

        current_transformation = result_icp.transformation
        print(current_transformation)

    return result_icp.transformation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--voxel_size', default=0.05, type=float)
    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.Debug)
    source = o3d.io.read_point_cloud(args.src)
    target = o3d.io.read_point_cloud(args.dst)
    voxel_size = args.voxel_size

    trans = multiscale_icp(source, target,
                           [voxel_size, voxel_size / 2.0, voxel_size / 4.0],
                           [50, 30, 14])
