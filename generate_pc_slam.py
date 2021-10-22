#!/usr/bin/env python3
import argparse
import glob
import open3d as o3d
from pathlib import Path
import numpy as np
import copy
import collections
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description='Generate pointclouds')
parser.add_argument("input_dir", help="Input directory.")
args = parser.parse_args()
assert (Path(args.input_dir).is_dir())


def get_rgbd_odom_images(input_dir: str):
    depth_imgs = sorted(glob.glob(input_dir + "/depth_*.png"))
    rgb_imgs = sorted(glob.glob(input_dir + "/rgb_*.png"))
    odom_infos = sorted(glob.glob(input_dir + "/odom_*.npy"))
    assert len(depth_imgs) == len(
        rgb_imgs), f"{len(depth_imgs)} vs {len(rgb_imgs)}"
    assert len(odom_infos) == len(
        depth_imgs), f"{len(odom_infos)} vs {len(depth_imgs)}"

    depth_imgs = [o3d.io.read_image(e) for e in depth_imgs]
    rgb_imgs = [o3d.io.read_image(e) for e in rgb_imgs]
    odom_infos = [np.load(e) for e in odom_infos]

    rgbd_odom_imgs = [
        (
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1000,  # Converts from mm to meters.
                depth_trunc=7,  # Truncate the depth image in meters.
                convert_rgb_to_intensity=False),
            odom) for rgb, depth, odom in zip(rgb_imgs, depth_imgs, odom_infos)
    ]

    return rgbd_odom_imgs


def make_intrinsic_matrix(input_dir):
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.width, camera_intrinsics.height = np.load(
        input_dir + "/depth_info_size.npy")
    camera_intrinsics.intrinsic_matrix = np.load(args.input_dir +
                                                 "/depth_info_K.npy")
    return camera_intrinsics


def make_point_cloud_rgbd_odom(depth_image: o3d.geometry.RGBDImage, odom: np.array,
                          intrinsics: o3d.camera.PinholeCameraIntrinsic):
    position = odom[:3]
    orientation = R.from_quat(odom[3:]).as_matrix()

    pc = o3d.geometry.PointCloud.create_from_rgbd_image(
        depth_image, intrinsics)
    #pc = pc.remove_non_finite_points().remove_statistical_outlier(10, 2)[0]
    pc_to_robot_frame = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    pc.rotate(pc_to_robot_frame, [0, 0, 0])
    # pc.translate(-odom_pose)
    # pc.rotate(odom_orientation, [0, 0, 0])
    return pc, depth_image, position, orientation


def make_origin_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.1)
    m.paint_uniform_color([0, 0, 0])
    return m


def print_extremes(pc: o3d.geometry.PointCloud):
    points = np.asarray(pc.points)
    for idx, name in [(0, 'x'), (1, 'y'), (2, 'z')]:
        print(name, ":", np.min(points[:, idx]), np.max(points[:, idx]))


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def transform_world_frame(camera_intrinsics, pc, rgbd, position, orientation, init_position,
                          init_orientation, prior_pc, prior_rgbd):
    pc = copy.deepcopy(pc)
    d_pos = position - init_position
    # Gets the delta in orientation between the two matrices
    d_rot = orientation @ init_orientation.T

    print(d_rot.shape)

    pc.rotate(d_rot, [0, 0, 0])
    pc.translate(d_pos)
    if prior_pc is None:
        return pc

    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.odometry.compute_rgbd_odometry(
         rgbd, prior_rgbd, camera_intrinsics,
         odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)



    # pc.transform(color_multiscale_icp(pc, prior_pc))
    return pc


def merge_pcs(pcs):
    if len(pcs) <= 0:
        return None
    out_pc = o3d.geometry.PointCloud()
    for pc in pcs:
        out_pc += pc
    return out_pc


camera_intrinsics = make_intrinsic_matrix(args.input_dir)
rgbd_odom_imgs = get_rgbd_odom_images(args.input_dir)
point_cloud_odoms = [
    make_point_cloud_rgbd_odom(rgbd, odom, camera_intrinsics)
    for rgbd, odom in rgbd_odom_imgs
]

# o3d.utility.set_verbosity_level(o3d.utility.Debug)
viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name="PointCloud Viewer")
_, init_position, init_orientation = point_cloud_odoms[0]
prior_pc_rgbd = None, None
for idx, (pc, rgbd, position,
          orentation) in enumerate(point_cloud_odoms[60:100]):  #[97:100]
    pc = transform_world_frame(camera_intrinsics, pc, rgbd, position, orentation, init_position,
                               init_orientation, *prior_pc_rgbd)
    viewer.add_geometry(pc)
    prior_pc_rgbd = copy.deepcopy(pc), rgbd
    print("idx", idx)
viewer.add_geometry(make_origin_sphere())
view = viewer.get_view_control()
view.set_lookat([0, 0, 0])
view.set_front([-1, 0, 0])
view.set_up([0, 0, 1])
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
opt.point_size = 0.01
viewer.run()
viewer.destroy_window()

# o3d.visualization.draw_geometries([point_clouds[0]],
#                                   lookat=[0, 0, 0],
#                                   up=[0, 0, 1],
#                                   front=[1, 0, 0],
#                                   zoom=1)
