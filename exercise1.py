import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import pi

BODY_MASSES = {
    "hat": 32.413,
    "pelvis": 11.15,
    "thigh": 8.806,
    "shank": 3.51,
    "foot": 1.20,
}

BODY_LENGTHS = {
    "thigh": 0.4009,
    "shank": 0.43,
    "foot": 0.1627,
}

COM_TO_JOINT = [
    ("pelvis", "hat", (0.0, 0.127, 0.0)),
    ("pelvis", "thigh", (0.0, -0.07, 0.0935)),
    ("hat", "pelvis", (0.0, -0.3202, 0.0)),
    ("thigh", "pelvis", (0.002, 0.1715, 0.0)),
    ("shank", "thigh", (0.0, 0.1862, 0.0)),
    ("foot", "shank", (-0.0359, 0.0513, 0.0055)),
]

ROTATIONS = {
    "hat": 60.0,
    "pelvis": 60.0,
    "thigh": 150.0,
    "shank": 60.0,
    "foot": 150.0
}


def unit_vector(vector):
    return vector / norm(vector)


def angle_of(v1):
    v1_u = unit_vector(v1)
    return np.arccos(np.clip(np.dot(v1_u, (1.0, 0.0)), -1.0, 1.0))


def sort_joints(joints):
    if joints[0][1] > joints[1][1]:
        joints.reverse()


def get_joints(part):
    com_to_joints = list(map(lambda data: data[2], filter(lambda data: data[0] == part, COM_TO_JOINT)))
    if len(com_to_joints) == 1:
        if part == "hat":
            com_to_joints.append((0.0, 0.0, 0.0))
        else:
            part_length = BODY_LENGTHS[part]
            joint = com_to_joints[0]
            used_length = norm(joint)
            if part != "foot":
                com_to_joints.append(tuple(np.multiply(joint, -(part_length - used_length) / used_length)))
            else:
                z = -joint[2]
                x = np.sqrt(np.square(norm((part_length, joint[1])) - norm((joint[0], joint[2]))) - np.square(z))
                com_to_joints.append((x, 0.0, z))
    sort_joints(com_to_joints)
    return com_to_joints


def get_rotated(joints, theta):
    """ Rotate joint vectors such that the angle between the part and the x-axis is theta """
    joint_to_joint_vec = np.subtract(joints[1], joints[0])
    theta_r = theta / 180.0 * pi - angle_of(joint_to_joint_vec[[0, 1]])
    rot = np.array([[np.cos(theta_r), -np.sin(theta_r), 0], [np.sin(theta_r), np.cos(theta_r), 0], [0, 0, 1]])
    rotated = list(map(lambda vec: tuple(np.matmul(rot, vec)), joints))
    sort_joints(rotated)
    return rotated


def get_part_joints(part):
    return part, get_rotated(get_joints(part), ROTATIONS[part])


def get_absolute_joints_and_coms(all_joints):
    absolute_joints = [(0.0, 0.0, 0.0)]
    coms_with_masses = []
    for i in range(len(all_joints) - 1, -1, -1):
        part, joints = all_joints[i]
        joint_start = absolute_joints[-1]
        coms_with_masses.append([BODY_MASSES[part]] + list(np.add(np.multiply(joints[0], -1), joint_start)))
        absolute_joints.append(tuple(np.add(np.subtract(joints[1], joints[0]), joint_start)))
    return absolute_joints, coms_with_masses


def get_cog(coms_with_masses):
    cwm = np.array(coms_with_masses)
    masses = cwm[:, 0]
    coms = cwm[:, [1, 2]]
    return tuple(np.dot(masses, coms) / np.sum(masses))


def solve():
    all_joints = list(map(get_part_joints, BODY_MASSES.keys()))
    joints, coms_with_masses = get_absolute_joints_and_coms(all_joints)
    print("center of mass: ", get_cog(coms_with_masses))
    plot(joints)


def plot(joints):
    for i in range(0, len(joints) - 1):
        origin = np.array(joints[i])[[0, 1]]
        direction = np.subtract(np.array(joints[i + 1])[[0, 1]], origin)
        plt.arrow(origin[0], origin[1], direction[0], direction[1])
        plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    solve()
