import numpy as np


"""
Taken heavily from https://github.com/orangeduck/Motion-Matching/blob/main/resources/quat.py
"""

def inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    # Hamilton product: https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    # Cross product of the imaginary part of the quaternion and the vector
    t = 2.0 * np.cross(q[..., 1:], x)
    # q[..., 0] -> reshape quaternion array to be scalar part
    #   -> [..., np.newaxis] adds new dimension at the end: https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis
    #       -> Note to self: could also do q[...,0:1] - would be the same. in source repo (see top), author uses that trick for _fast_cross
    #   -> this effictively multiplies t by the real part of each quaternion
    # np.cross(q[..,1:], t) -> cross product of real part of quaternion and t
    return x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

"""
From: Mode-adaptive neural networks for quadruped motion control - https://dl.acm.org/doi/10.1145/3197517.3201366
'We represent the joint rotations by the relative forward and upward vectors in
order to avoid quaternion interpolation issues by the neural network during training.'
"""
def two_axis(q):
    forward = np.array([0, 0, 1.0])
    up = np.array([0, 1.0, 0])
    forward_vecs = mul_vec(q, forward)
    up_vecs = mul_vec(q, up)
    # concatenate (n, 3) and (n, 3) vec to get (n, 6) vec
    return np.concatenate([forward_vecs, up_vecs], axis=-1)
