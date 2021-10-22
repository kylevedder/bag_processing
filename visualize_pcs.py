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


def make_origin_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.1)
    m.paint_uniform_color([0, 0, 0])
    return m


def quat_to_mat(quat):
    return R.from_quat(quat).as_matrix()


def get_odom_delta(next, prior):
    return next @ prior.T


def normalize_start(pos, rot, start_pos, start_rot):
    pos = pos - start_pos
    rot = get_odom_delta(rot, start_rot)
    return pos, rot


def to_homogenious_matrix(pos, rot):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = rot
    return mat


def get_intrinsic_matrix(input_dir):
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.width, camera_intrinsics.height = np.load(
        input_dir + "/depth_info_size.npy")
    camera_intrinsics.intrinsic_matrix = np.load(args.input_dir +
                                                 "/depth_info_K.npy")
    return camera_intrinsics


def get_rgbd_odoms(input_dir: str):
    depth_imgs = sorted(glob.glob(input_dir + "/depth_*.png"))
    rgb_imgs = sorted(glob.glob(input_dir + "/rgb_*.png"))
    associations = open(input_dir + "/associations.txt").readlines()
    odom_infos = open(input_dir + "/CameraTrajectory.txt").readlines()
    assert len(depth_imgs) == len(
        rgb_imgs), f"{len(depth_imgs)} vs {len(rgb_imgs)}"

    odom_infos = [np.array(e.split(' '), dtype=np.float64) for e in odom_infos]
    odom_times = set([e[0] for e in odom_infos])
    association_times = [float(e.split(' ')[0]) for e in associations]

    depth_imgs = [o3d.io.read_image(e) for idx, e in enumerate(depth_imgs) if association_times[idx] in odom_times]
    rgb_imgs = [o3d.io.read_image(e) for idx, e in enumerate(rgb_imgs) if association_times[idx] in odom_times]
    print(f"After filter found {len(depth_imgs)} depth and {len(rgb_imgs)} rgb")
    
    odom_infos = [e[1:] for e in odom_infos]
    odom_infos = [(e[:3], quat_to_mat(e[3:])) for e in odom_infos]
    odom_infos = [normalize_start(*e, *odom_infos[0]) for e in odom_infos]
    odom_infos = [to_homogenious_matrix(*e) for e in odom_infos]

    print(f"len(odom_infos) {len(odom_infos)} len(depth_imgs) {len(depth_imgs)} len(rgb_imgs) {len(rgb_imgs)}")

    rgbd_odoms = [
        (
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1000,  # Converts from mm to meters.
                depth_trunc=7,  # Truncate the depth image in meters.
                convert_rgb_to_intensity=False),
            odom) for rgb, depth, odom in zip(rgb_imgs, depth_imgs, odom_infos)
    ]

    print(f"Returning {len(rgbd_odoms)} rgbd_odoms")

    return rgbd_odoms


def rgbd_odom_to_pc(rgbd, odom, camera_intrinsics):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, camera_intrinsics)
    pc_to_robot_frame = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    pc.transform(odom)
    pc.rotate(pc_to_robot_frame, [0, 0, 0])
    return pc


camera_intrinsics = get_intrinsic_matrix(args.input_dir)
rgbd_odoms = get_rgbd_odoms(args.input_dir)

# o3d.utility.set_verbosity_level(o3d.utility.Debug)
viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name="PointCloud Viewer")

viewer.add_geometry(make_origin_sphere())

for idx, (rgbd, odom) in enumerate(rgbd_odoms):
    viewer.add_geometry(rgbd_odom_to_pc(rgbd, odom, camera_intrinsics))

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