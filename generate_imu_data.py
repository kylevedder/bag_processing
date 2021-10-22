#!/usr/bin/env python3
import argparse
from math import degrees
import numpy as np
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description='Generate pointclouds')
parser.add_argument("input_dir", help="Input directory.")
args = parser.parse_args()
assert (Path(args.input_dir).is_dir())


def quat_to_mat(quat):
    return R.from_quat(quat).as_matrix()


def get_rot_delta(next, prior):
    return next @ prior.T


def normalize_start(t, pos, rot, start_t, start_pos, start_rot):
    pos = pos - start_pos
    rot = get_rot_delta(rot, start_rot)
    return t, pos, rot


def compute_imu(odom_infos):
    times = [t for t, _, _ in odom_infos]
    # Rad / s
    ang_vel = [
        R.from_matrix(get_rot_delta(ang2, ang1)).as_euler('xyz', degrees=False)
        / (t2 - t1)
        for (t1, _, ang1), (t2, _, ang2) in zip(odom_infos, odom_infos[1:])
    ]
    ang_vel.insert(0, np.zeros((3, )))

    # m/s
    pos_vel_times = [((pos2 - pos1) / (t2 - t1), t2)
                     for (t1, pos1, _), (t2, pos2,
                                         _) in zip(odom_infos, odom_infos[1:])]
    pos_vel_times.insert(0, (np.zeros((3, )), odom_infos[0][0]))

    # m/s^2
    pos_acc = [(vel2 - vel1) / (t2 - t1)
               for (vel1, t1), (vel2, t2) in zip(pos_vel_times, pos_vel_times[1:])]
    pos_acc.insert(0, np.zeros((3, )))

    return times, pos_acc, ang_vel


odom_infos = sorted(glob.glob(args.input_dir + "/odom_*.npy"))
odom_infos = [np.load(e) for e in odom_infos]
odom_infos = [(e[0], e[1:4], quat_to_mat(e[4:])) for e in odom_infos]
odom_infos = [normalize_start(*e, *odom_infos[0]) for e in odom_infos]
times, pos_acc, ang_vel = compute_imu(odom_infos)

assert len(pos_acc) == len(ang_vel), f"{len(pos_acc)} vs {len(ang_vel)}"
assert len(pos_acc) == len(odom_infos), f"{len(pos_acc)} vs {len(odom_infos)}"

f = open(args.input_dir + "/imu_data.txt", 'w')
for t, acc, vel in zip(times, pos_acc, ang_vel):
    f.write("{} {} {} {} {} {} {}\n".format(t, *vel, *acc))
f.close()
