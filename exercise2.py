from math import pi
import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt

REVERSE_DISTAL = ["trunk", "head&neck"]

JOINTS_TO_MARKERS = {
    "r foot": ["r hfm", "r lm"],
    "r shank": ["r lm", "r le"],
    "r thigh": ["r le", "r gt"],
    "r forearm": ["r wrist", "r elbo"],
    "r upperarm": ["r elbo", "r shoulder"],
    "l foot": ["l hfm", "l lm"],
    "l shank": ["l lm", "l le"],
    "l thigh": ["l le", "l gt"],
    "l forearm": ["l wrist", "l elbow"],
    "l upperarm": ["l elbow", "l shoulder"],
    "pelvis": ["r asis", "l asis"],
    "trunk": ["r asis", "l asis"],
    "head&neck": ["forehead"],
}

BODY_WEIGHT = {
    "r foot": 0.0145,
    "r shank": 0.0465,
    "r thigh": 0.1,
    "r forearm": 0.016,
    "r upperarm": 0.028,
    "l foot": 0.0145,
    "l shank": 0.0465,
    "l thigh": 0.1,
    "l forearm": 0.016,
    "l upperarm": 0.028,
    "pelvis": 0.142,
    "trunk": 0.355,
    "head&neck": 0.081,
    # Computations as combinations from above
    "r leg": 0.0145 + 0.0465 + 0.1,
    "l leg": 0.0145 + 0.0465 + 0.1,
    "r arm": 0.016 + 0.028,
    "l arm": 0.016 + 0.028,
    "torso": 0.142 + 0.355,
    "legs": 2 * (0.0145 + 0.0465 + 0.1),
    "torso&arms": 0.142 + 0.355 + 2 * (0.016 + 0.028),
}

LENGTH_TO_DISTAL = {
    "l foot": 0.5,
    "l shank": 0.567,
    "l thigh": 0.567,
    "l forearm": 0.57,
    "l upperarm": 0.564,
    "r foot": 0.5,
    "r shank": 0.567,
    "r thigh": 0.567,
    "r forearm": 0.57,
    "r upperarm": 0.564,
    "pelvis": 0.895,
    "trunk": 0.31,
    "head&neck": 0.5,
}


def get_rotated(joint_coords, theta):
    """ Rotate joint vectors such that the angle between the part and the x-axis is theta """
    theta_r = (theta - 90.0) / 180.8 * pi
    rot = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
    rotated = list(map(lambda vec: tuple(np.matmul(rot, vec)), joint_coords))
    return rotated


def get_cog(coms_with_masses):
    cwm = np.array(coms_with_masses)
    masses = cwm[:, 0]
    coms = cwm[:, [1, 2]]
    return tuple(np.dot(masses, coms) / np.sum(masses))


def get_combined(joint_data):
    coms_with_masses = []
    joint_start = (0.0, 0.0)
    joint_end = joint_start
    for dz, length, angle, name in joint_data:
        joints = np.multiply(np.array([[0.0, -LENGTH_TO_DISTAL[name]], [0.0, 1.0 - LENGTH_TO_DISTAL[name]]]), length)
        if name in REVERSE_DISTAL:
            joints = np.flip(np.multiply(joints, -1), axis=0)
        joints = get_rotated(joints, angle)
        coms_with_masses.append([BODY_WEIGHT[name]] + list(np.add(np.multiply(joints[0], -1), joint_end)))
        joint_end = tuple(np.add(np.subtract(joints[1], joints[0]), joint_end))
    combined_cg = get_cog(coms_with_masses)
    return np.array([np.subtract(joint_start, combined_cg), np.subtract(joint_end, combined_cg)])


def get_combined_overlapping(overlapping_parts, adjust_bottom=False):
    """ Joins all provided parts (pair of joints) at top joint. If adjust=True, sets the points to the top &
    bottom-most joints """
    coms = []
    bottom_joint = overlapping_parts[0][1][0]
    for part, joints in overlapping_parts:
        coms.append([BODY_WEIGHT[part]] + list(np.multiply(joints[1], -1)))
        if adjust_bottom and joints[0][1] < bottom_joint[1]:
            bottom_joint = joints[0]
    combined_cg = get_cog(coms)
    return np.subtract([bottom_joint, (0.0, 0.0)], [combined_cg, combined_cg])


def get_absolute_joints_and_coms(all_joints):
    absolute_joints = [(0.0, 0.0)]
    coms_with_masses = []
    for part, joints in all_joints:
        joint_start = absolute_joints[-1]
        coms_with_masses.append([BODY_WEIGHT[part]] + list(np.add(np.multiply(joints[0], -1), joint_start)))
        absolute_joints.append(tuple(np.add(np.subtract(joints[1], joints[0]), joint_start)))
    return absolute_joints, coms_with_masses


def get_combined_by_parts(parts, dzs, lengths, angles, row):
    combined_data = list(map(lambda part: (dzs.iloc[row][part], lengths.iloc[row][part], angles.iloc[row][part], part),
                             parts))
    return get_combined(combined_data)


def get_dz_data(raw_z_data):
    df = pd.DataFrame(raw_z_data.iloc[:, :2])
    for joint in JOINTS_TO_MARKERS.keys():
        markers = JOINTS_TO_MARKERS[joint]
        df[joint] = raw_z_data[markers].mean(axis=1)
    df = df - df.iloc[0].values.squeeze()
    return df


def get_data():
    df = read_csv("data.csv", header=0)
    raw_z_data = df.iloc[:, :19]
    raw_z_data.columns = (x.replace('(mm)', '').strip().lower() for x in raw_z_data.columns)
    z_data = get_dz_data(raw_z_data)
    length_data = df.iloc[:, 19:32]
    length_data.columns = (x.strip().lower() for x in length_data.columns)
    angle_data = df.iloc[:, 32:-1]
    angle_data.columns = (x.replace('.1', '').strip().lower() for x in angle_data.columns)
    return z_data, length_data, angle_data


def plot(joints, coms):
    plt.clf()
    for i in range(0, len(joints) - 1):
        origin = np.array(joints[i])[[0, 1]]
        direction = np.subtract(np.array(joints[i + 1])[[0, 1]], origin)
        plt.arrow(origin[0], origin[1], direction[0], direction[1])
        plt.gca().set_aspect('equal', adjustable='box')
    for com in coms:
        plt.plot(com[1], com[2], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.draw()
    plt.pause(0.02)


def solve_row(row, dzs, lengths, angles):
    r_leg = get_combined_by_parts(["r foot", "r shank", "r thigh"], dzs, lengths, angles, row)
    l_leg = get_combined_by_parts(["l foot", "l shank", "l thigh"], dzs, lengths, angles, row)
    r_arm = get_combined_by_parts(["r forearm", "r upperarm"], dzs, lengths, angles, row)
    l_arm = get_combined_by_parts(["l forearm", "l upperarm"], dzs, lengths, angles, row)
    torso = get_combined_by_parts(["pelvis", "trunk"], dzs, lengths, angles, row)
    head = get_combined_by_parts(["head&neck"], dzs, lengths, angles, row)
    legs = get_combined_overlapping([("r leg", r_leg), ("l leg", l_leg)], adjust_bottom=True)
    torso_and_arms = get_combined_overlapping([("torso", torso), ("r arm", r_arm), ("l arm", l_arm)])
    absolute_joints, coms = get_absolute_joints_and_coms(
        [("legs", legs), ("torso&arms", torso_and_arms), ("head&neck", head)])
    return get_cog(coms), coms, absolute_joints


def solve():
    dzs, lengths, angles = get_data()
    coms = []
    for i in range(len(dzs)):
        com, sub_coms, joints = solve_row(i, dzs, lengths, angles)
        print(com)
        coms.append(com)
        plot(joints, sub_coms)
    plt.show()


if __name__ == '__main__':
    solve()
