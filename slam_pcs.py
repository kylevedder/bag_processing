#!/usr/bin/env python3
import argparse
import glob
import open3d as o3d
from pathlib import Path
import numpy as np
import copy
import collections
from scipy.spatial.transform import Rotation as R

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Generate pointclouds')
parser.add_argument("input_dir", help="Input directory.")
parser.add_argument("--icp", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
args = parser.parse_args()
assert (Path(args.input_dir).is_dir())


def make_origin_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.1)
    m.paint_uniform_color([0, 0, 0])
    return m


def quat_to_mat(quat):
    return R.from_quat(quat).as_matrix()


def get_rot_delta(next, prior):
    return next @ prior.T


def normalize_start(pos, rot, start_pos, start_rot):
    pos = pos - start_pos
    rot = get_rot_delta(rot, start_rot)
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
    odom_infos = sorted(glob.glob(input_dir + "/odom_*.npy"))
    assert len(depth_imgs) == len(
        rgb_imgs), f"{len(depth_imgs)} vs {len(rgb_imgs)}"
    assert len(odom_infos) == len(
        depth_imgs), f"{len(odom_infos)} vs {len(depth_imgs)}"
    print(f"Found {len(depth_imgs)} images")

    depth_imgs = [o3d.io.read_image(e) for e in depth_imgs]
    rgb_imgs = [o3d.io.read_image(e) for e in rgb_imgs]
    odom_infos = [np.load(e) for e in odom_infos]
    odom_infos = [(e[1:4], quat_to_mat(e[4:])) for e in odom_infos]
    odom_infos = [normalize_start(*e, *odom_infos[0]) for e in odom_infos]
    odom_infos = [to_homogenious_matrix(*e) for e in odom_infos]

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

    return rgbd_odoms


def rgbd_odom_to_pc(rgbd, odom, camera_intrinsics):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, camera_intrinsics)
    pc_to_robot_frame = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    pc.rotate(pc_to_robot_frame, [0, 0, 0])
    pc.transform(odom)
    return pc


camera_intrinsics = get_intrinsic_matrix(args.input_dir)
print(camera_intrinsics)
print(camera_intrinsics.intrinsic_matrix)
rgbd_odoms = get_rgbd_odoms(args.input_dir)#[80:90]

# o3d.utility.set_verbosity_level(o3d.utility.Debug)
viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name="PointCloud Viewer")

viewer.add_geometry(make_origin_sphere())
viewer.add_geometry(rgbd_odom_to_pc(*rgbd_odoms[0], camera_intrinsics))

for idx, ((rgbd_t, odom_t),
          (rgbd_tp1, odom_tp1)) in enumerate(zip(rgbd_odoms, rgbd_odoms[1:])):
    print("idx:", idx)
    odom_motion_delta = odom_tp1 @ np.linalg.inv(odom_t)
    slam_motion_delta = odom_motion_delta
    if args.icp:
        odom_opt = o3d.pipelines.odometry.OdometryOption(
            min_depth=0.5,
            max_depth=7,
            max_depth_diff=0.005,
            #iteration_number_per_pyramid_level=o3d.utility.IntVector([1, 1, 1])
            )
        success_hybrid_term, slam_motion_delta, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_t, rgbd_tp1, camera_intrinsics, odom_motion_delta,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odom_opt)
        if not success_hybrid_term:
            print("visual odom failed")
            slam_motion_delta = odom_motion_delta
    
    
    corrected_odom_tp1 = slam_motion_delta @ odom_t
    viewer.add_geometry(
        rgbd_odom_to_pc(rgbd_tp1, corrected_odom_tp1, camera_intrinsics))
    rgbd_odoms[idx + 1] = (rgbd_tp1, corrected_odom_tp1)

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