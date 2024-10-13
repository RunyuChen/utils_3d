# transform quaternion to R matrix
def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    - quaternion: A tensor of shape (..., 4) representing quaternions.

    Returns:
    - A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Ensure quaternion is of float type for computation
    quaternion = quaternion.float()

    # Normalize the quaternion to unit length
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    # Assemble the rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)], dim=-1),
        torch.stack([    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)], dim=-1),
        torch.stack([    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)

    return R
