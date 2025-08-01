import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import re

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def axis_angle_to_rotmat(axis_angle):
    return R.from_rotvec(axis_angle).as_matrix()

def slerp_rotation(prev_rot, next_rot, alpha):
    """Perform spherical linear interpolation (SLERP) between two rotations.
    
    Args:
        prev_rot: scipy.spatial.transform.Rotation object for previous rotation
        next_rot: scipy.spatial.transform.Rotation object for next rotation
        alpha: interpolation factor (0.0 = prev_rot, 1.0 = next_rot)
    
    Returns:
        scipy.spatial.transform.Rotation object for interpolated rotation
    """
    # Convert to quaternions for SLERP
    prev_quat = prev_rot.as_quat()
    next_quat = next_rot.as_quat()
    
    # Ensure shortest path (flip quaternion if needed)
    if np.dot(prev_quat, next_quat) < 0:
        next_quat = -next_quat
    
    # SLERP interpolation
    interpolated_quat = (1 - alpha) * prev_quat + alpha * next_quat
    interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat)
    
    # Convert back to rotation
    return R.from_quat(interpolated_quat)

def find_nearest_valid_frames(idx, mano_indices, max_distance=10):
    """Find the nearest valid frames before and after the current frame."""
    current_idx = int(idx)
    valid_indices = [int(k) for k in mano_indices.keys()]
    valid_indices.sort()
    
    # Find the nearest valid frame before current
    prev_valid = None
    for i in range(current_idx - 1, max(current_idx - max_distance, min(valid_indices) - 1), -1):
        if str(i).zfill(5) in mano_indices:
            prev_valid = i
            break
    
    # Find the nearest valid frame after current
    next_valid = None
    for i in range(current_idx + 1, min(current_idx + max_distance, max(valid_indices) + 1)):
        if str(i).zfill(5) in mano_indices:
            next_valid = i
            break
    
    return prev_valid, next_valid

def interpolate_hand_pose(prev_frame, next_frame, current_frame, mano_indices, mano_folder, hand_type):
    """Interpolate hand pose between two valid frames."""
    if prev_frame is None and next_frame is None:
        return np.tile(np.eye(3), (15, 1, 1))  # Default pose if no valid frames
    
    if prev_frame is None:
        # Use next frame pose
        mano_path = os.path.join(mano_folder, mano_indices[str(next_frame).zfill(5)])
        mano = load_pkl(mano_path)
        mano_id = list(mano.keys())[0]
        if mano[mano_id][hand_type] is not None:
            hand_pose = mano[mano_id][hand_type]['hand_pose']
            return R.from_rotvec(hand_pose).as_matrix()
        return np.tile(np.eye(3), (15, 1, 1))
    
    if next_frame is None:
        # Use prev frame pose
        mano_path = os.path.join(mano_folder, mano_indices[str(prev_frame).zfill(5)])
        mano = load_pkl(mano_path)
        mano_id = list(mano.keys())[0]
        if mano[mano_id][hand_type] is not None:
            hand_pose = mano[mano_id][hand_type]['hand_pose']
            return R.from_rotvec(hand_pose).as_matrix()
        return np.tile(np.eye(3), (15, 1, 1))
    
    # Interpolate between prev and next frames
    mano_prev_path = os.path.join(mano_folder, mano_indices[str(prev_frame).zfill(5)])
    mano_next_path = os.path.join(mano_folder, mano_indices[str(next_frame).zfill(5)])
    
    mano_prev = load_pkl(mano_prev_path)
    mano_next = load_pkl(mano_next_path)
    
    mano_prev_id = list(mano_prev.keys())[0]
    mano_next_id = list(mano_next.keys())[0]
    
    # Get hand poses from both frames
    prev_hand_pose = None
    next_hand_pose = None
    
    if mano_prev[mano_prev_id][hand_type] is not None:
        prev_hand_pose = mano_prev[mano_prev_id][hand_type]['hand_pose']
    
    if mano_next[mano_next_id][hand_type] is not None:
        next_hand_pose = mano_next[mano_next_id][hand_type]['hand_pose']
    
    # If both are None, return default pose
    if prev_hand_pose is None and next_hand_pose is None:
        return np.tile(np.eye(3), (15, 1, 1))
    
    # If only one is None, use the other
    if prev_hand_pose is None:
        return R.from_rotvec(next_hand_pose).as_matrix()
    if next_hand_pose is None:
        return R.from_rotvec(prev_hand_pose).as_matrix()
    
    # Interpolate between the two poses
    total_distance = float(next_frame - prev_frame)
    current_distance = float(current_frame - prev_frame)
    alpha = current_distance / total_distance
    
    # Convert to rotation matrices for interpolation
    prev_rotmat = R.from_rotvec(prev_hand_pose).as_matrix()  # (15, 3, 3)
    next_rotmat = R.from_rotvec(next_hand_pose).as_matrix()  # (15, 3, 3)
    
    # Interpolate each joint rotation using SLERP
    interpolated_rotmat = np.zeros_like(prev_rotmat)
    for i in range(15):
        prev_rot = R.from_matrix(prev_rotmat[i])
        next_rot = R.from_matrix(next_rotmat[i])
        
        # Perform SLERP interpolation
        interpolated_rot = slerp_rotation(prev_rot, next_rot, alpha)
        interpolated_rotmat[i] = interpolated_rot.as_matrix()
    
    return interpolated_rotmat

# ASSUMES FILES AND IN FORMAT mano_XXXXX.pkl and smpl_params_XXXXX.pkl
def main(smpl_folder: str, mano_folder: str, output_folder: str, max_interpolation_distance: int = 10):
    # Find all mano_XXXXX.pkl and smplx_combined_XXXXX.pkl files
    mano_files = [f for f in os.listdir(mano_folder) if re.match(r'mano_\d{5}\.pkl', f)]
    smpl_files = [f for f in os.listdir(smpl_folder) if re.match(r'smplx_combined_\d{5}\.pkl', f)]
    
    # print found files
    print(f'Found {len(mano_files)} mano files')
    print(f'Found {len(smpl_files)} smplx_combined files')

    # Extract indices
    mano_indices = {re.findall(r'\d{5}', f)[0]: f for f in mano_files}
    smpl_indices = {re.findall(r'\d{5}', f)[0]: f for f in smpl_files}
    # Iterate over all SMPL indices (not just common ones)
    all_smpl_indices = sorted(smpl_indices.keys())
    print(f'Processing {len(all_smpl_indices)} smplx_combined files (output will be created for each)')

    os.makedirs(output_folder, exist_ok=True)

    for idx in all_smpl_indices:
        smplx_path = os.path.join(smpl_folder, smpl_indices[idx])
        smplx_data = load_pkl(smplx_path)
        smplx_id = list(smplx_data.keys())[0]

        # Extract body_pose (first 21 joints)
        body_pose = smplx_data[smplx_id]['smplx_params']['body_pose'][:21]  # shape (21, 3, 3)

        # Initialize hand poses
        left_hand_pose = None
        right_hand_pose = None

        # If corresponding MANO file exists, use its hand poses
        if idx in mano_indices:
            mano_path = os.path.join(mano_folder, mano_indices[idx])
            mano = load_pkl(mano_path)
            mano_id = list(mano.keys())[0]
            if mano[mano_id]['left_hand'] is not None:
                left_hand_pose = mano[mano_id]['left_hand']['hand_pose']  # shape (15, 3)
                left_hand_pose = R.from_rotvec(left_hand_pose).as_matrix()
            if mano[mano_id]['right_hand'] is not None:
                right_hand_pose = mano[mano_id]['right_hand']['hand_pose']  # shape (15, 3)
                right_hand_pose = R.from_rotvec(right_hand_pose).as_matrix()
        
        # If MANO file doesn't exist or hands are not detected, interpolate
        if idx not in mano_indices or left_hand_pose is None:
            prev_valid, next_valid = find_nearest_valid_frames(idx, mano_indices, max_interpolation_distance)
            left_hand_pose = interpolate_hand_pose(prev_valid, next_valid, int(idx), mano_indices, mano_folder, 'left_hand')
        
        if idx not in mano_indices or right_hand_pose is None:
            prev_valid, next_valid = find_nearest_valid_frames(idx, mano_indices, max_interpolation_distance)
            right_hand_pose = interpolate_hand_pose(prev_valid, next_valid, int(idx), mano_indices, mano_folder, 'right_hand')

        # Identity for jaw (3 joints) (3, 3, 3)
        jaw_identity = np.tile(np.eye(3), (3, 1, 1))

        # Update the SMPL-X parameters with separate fields
        smplx_params = smplx_data[smplx_id]['smplx_params']
        smplx_params['body_pose'] = body_pose
        if 'jaw_pose' not in smplx_params or smplx_params['jaw_pose'] is None:
            smplx_params['jaw_pose'] = jaw_identity
        smplx_params['left_hand_pose'] = left_hand_pose
        smplx_params['right_hand_pose'] = right_hand_pose
        if 'combined_body_pose' in smplx_params:
            del smplx_params['combined_body_pose']

        # Create the new structure
        new_smplx_data = {
            smplx_id: {
                'smplx_params': smplx_params
            }
        }

        out_path = os.path.join(output_folder, f'smplx_combined_{idx}.pkl')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_pkl(new_smplx_data, out_path)
        
        # Determine interpolation status for logging
        mano_found = idx in mano_indices
        left_interpolated = idx not in mano_indices or (mano_found and left_hand_pose is None)
        right_interpolated = idx not in mano_indices or (mano_found and right_hand_pose is None)
        
        interpolation_status = []
        if left_interpolated:
            interpolation_status.append("left_hand_interpolated")
        if right_interpolated:
            interpolation_status.append("right_hand_interpolated")
        
        status_str = f"MANO: {'found' if mano_found else 'not_found'}"
        if interpolation_status:
            status_str += f", {', '.join(interpolation_status)}"
        
        print(
            f"Processed {smplx_path}. {status_str}. Saved to {out_path}. "
            f"body_pose shape: {body_pose.shape}, "
            f"jaw_pose shape: {jaw_identity.shape}, "
            f"left_hand_pose shape: {left_hand_pose.shape}, "
            f"right_hand_pose shape: {right_hand_pose.shape}"
        )
    print("Done!")

if __name__ == '__main__':
    tyro.cli(main)