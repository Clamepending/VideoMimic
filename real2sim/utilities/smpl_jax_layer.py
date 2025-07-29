# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""SMPL model, implemented in JAX.

Very little of it is specific to SMPL. This could very easily be adapted for other models in SMPL family.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, cast, Optional

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from einops import einsum
from jax import Array
from jax import numpy as jnp


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@jdc.pytree_dataclass
class SmplModel:
    """The SMPL human body model."""

    faces: Array
    """Vertex indices for mesh faces."""
    J_regressor: Array
    """Linear map from vertex to joint positions.
    23+1 body joints """
    parent_indices: Array
    """Defines kinematic tree. Index of -1 signifies that a joint is defined
    relative to the root."""
    weights: Array
    """LBS weights."""
    posedirs: Array
    """Pose blend shape bases."""
    v_template: Array
    """Canonical mesh verts."""
    shapedirs: Array
    """Shape bases."""

    @staticmethod
    def load(pickle_path: Path) -> SmplModel:
        # smpl_params: dict[str, onp.ndarray] = onp.load(npz_path, allow_pickle=True)

        with open(pickle_path, 'rb') as smpl_file:
            # smpl_params = Struct(**pickle.load(smpl_file, encoding='latin1'))
            smpl_params = pickle.load(smpl_file, encoding='latin1')

        # assert smpl_params["bs_style"].item() == b"lbs"
        # assert smpl_params["bs_type"].item() == b"lrotmin"
        valid_keys = ["f", "J_regressor", "kintree_table", "weights", "posedirs", "v_template", "shapedirs"]
        smpl_params["shapedirs"] = onp.array(smpl_params["shapedirs"], dtype=onp.float32)
        smpl_params["J_regressor"] = onp.array(smpl_params["J_regressor"].toarray(), dtype=onp.float32)
        
        smpl_params = {k: _normalize_dtype(v) for k, v in smpl_params.items() if k in valid_keys}

        return SmplModel(
            faces=jnp.array(smpl_params["f"]),
            J_regressor=jnp.array(smpl_params["J_regressor"]),
            parent_indices=jnp.array(smpl_params["kintree_table"][0][1:] - 1),
            weights=jnp.array(smpl_params["weights"]),
            posedirs=jnp.array(smpl_params["posedirs"]),
            v_template=jnp.array(smpl_params["v_template"]),
            shapedirs=jnp.array(smpl_params["shapedirs"]),
        )

    def with_shape(
        self, betas: Array | onp.ndarray
    ) -> SmplShaped:
        """Compute a new body model, with betas applied. betas vector should
        have shape up to (10,)."""
        num_betas = betas.shape[-1]
        assert num_betas <= 10
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_with_shape,
            "joints verts, ... verts xyz -> ... joints xyz",
        )
        root_offset = root_and_joints_pred[..., 0:1, :]
        return SmplShaped(
            body_model=self,
            verts_zero=verts_with_shape - root_offset,
            joints_zero=root_and_joints_pred[..., 1:, :] - root_offset,
            t_parent_joint=root_and_joints_pred[..., 1:, :]
            - root_and_joints_pred[..., self.parent_indices + 1, :],
        )


@jdc.pytree_dataclass
class SmplShaped:
    """The SMPL body model with a body shape applied."""

    body_model: SmplModel
    verts_zero: Array
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Array
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Array
    """Position of each shaped body joint relative to its parent. Does not
    include root."""

    def with_pose_decomposed(
        self,
        T_world_root: Array | onp.ndarray,
        body_quats: Array | onp.ndarray,
    ) -> SmplShapedAndPosed:
        """Pose our SMPL body model. Returns a set of joint and vertex outputs."""

        local_quats = broadcasting_cat(
            cast(list[jax.Array], [body_quats]),
            axis=0,
        )
        assert local_quats.shape[-2:] == (23, 4)
        return self.with_pose(T_world_root, local_quats)

    def with_pose(
        self,
        T_world_root: Array | onp.ndarray,
        local_quats: Array | onp.ndarray,
    ) -> SmplShapedAndPosed:
        """Pose our SMPL body model. Returns a set of joint and vertex outputs."""

        # Forward kinematics.
        # assert local_quats.shape == (23, 4), local_quats.shape
        parent_indices = self.body_model.parent_indices
        (num_joints,) = parent_indices.shape[-1:]
        num_active_joints, _ = local_quats.shape[-2:]
        assert local_quats.shape[-1] == 4
        assert num_active_joints <= num_joints
        assert self.t_parent_joint.shape[-2:] == (num_joints, 3)

        # Get relative transforms.
        Ts_parent_child = broadcasting_cat(
            [local_quats, self.t_parent_joint[..., :num_active_joints, :]], axis=-1
        )
        assert Ts_parent_child.shape[-2:] == (num_active_joints, 7)

        # Compute one joint at a time.
        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                parent_indices[i] == -1,
                T_world_root,
                Ts_world_joint[..., parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros_like(Ts_parent_child),
        )
        assert Ts_world_joint.shape[-2:] == (num_active_joints, 7)

        return SmplShapedAndPosed(
            shaped_model=self,
            T_world_root=T_world_root,  # type: ignore
            local_quats=local_quats,  # type: ignore
            Ts_world_joint=Ts_world_joint,
        )

    def get_T_head_cpf(self) -> Array:
        """Get the central pupil frame with respect to the head (joint 14). This
        assumes that we're using the SMPL model."""

        assert self.verts_zero.shape[-2:] == (6890, 3), "Not using SMPL model!"
        right_eye = (
            self.verts_zero[..., 6260, :] + self.verts_zero[..., 6262, :]
        ) / 2.0
        left_eye = (self.verts_zero[..., 2800, :] + self.verts_zero[..., 2802, :]) / 2.0

        # CPF is between the two eyes.
        cpf_pos_wrt_head = (right_eye + left_eye) / 2.0 - self.joints_zero[..., 14, :]

        return broadcasting_cat([jaxlie.SO3.identity().wxyz, cpf_pos_wrt_head], axis=-1)


@jdc.pytree_dataclass
class SmplShapedAndPosed:
    shaped_model: SmplShaped
    """Underlying shaped body model."""

    T_world_root: Array
    """Root coordinate frame."""

    local_quats: Array
    """Local joint orientations."""

    Ts_world_joint: Array
    """Absolute transform for each joint. Does not include the root."""

    def with_new_T_world_root(
        self, T_world_root: Array
    ) -> SmplShapedAndPosed:
        return SmplShapedAndPosed(
            shaped_model=self.shaped_model,
            T_world_root=T_world_root,
            local_quats=self.local_quats,
            Ts_world_joint=(
                jaxlie.SE3(T_world_root[..., None, :])
                @ jaxlie.SE3(self.T_world_root[..., None, :]).inverse()
                @ jaxlie.SE3(self.Ts_world_joint)
            ).parameters(),
        )

    def lbs(self) -> SmplMesh:
        assert (
            self.local_quats.shape[0]
            == self.shaped_model.body_model.parent_indices.shape[0]
        ), "It looks like only a partial set of joint rotations was passed into `with_pose()`. We need all of them for LBS."

        # Linear blend skinning with a pose blend shape.
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (jaxlie.SO3(self.local_quats).as_matrix() - jnp.eye(3)).flatten(),
            "verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )
        verts_transformed = einsum(
            broadcasting_cat(
                [
                    # (*, 1, 3, 4)
                    jaxlie.SE3(self.T_world_root).as_matrix()[..., None, :3, :],
                    # (*, 51, 3, 4)
                    jaxlie.SE3(self.Ts_world_joint).as_matrix()[..., :3, :],
                ],
                axis=0,
            ),
            self.shaped_model.body_model.weights,
            jnp.pad(
                verts_with_blend[:, None, :]
                - jnp.concatenate(
                    [
                        jnp.zeros((1, 1, 3)),  # Root joint.
                        self.shaped_model.joints_zero[None, :, :],
                    ],
                    axis=1,
                ),
                ((0, 0), (0, 0), (0, 1)),
                constant_values=1.0,
            ),
            "joints_p1 i j, ... verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )

        return SmplMesh(
            posed_model=self,
            verts=verts_transformed,
            faces=self.shaped_model.body_model.faces,
        )


@jdc.pytree_dataclass
class SmplMesh:
    posed_model: SmplShapedAndPosed

    verts: Array
    """Vertices for mesh."""

    faces: Array
    """Faces for mesh."""

@jdc.pytree_dataclass
class SmplxModel:
    """SMPL-X human body model with hands and facial expressions."""
    
    # Core mesh and kinematic structure
    faces: Array
    """Vertex indices for mesh faces."""
    
    J_regressor: Array
    """Linear map from vertex to joint positions.
    21+1 body joints """
    
    parent_indices: Array
    """Defines kinematic tree. Index of -1 signifies that a joint is defined
    relative to the root."""
    
    weights: Array
    """LBS weights."""
    
    # Deformation components
    posedirs: Array
    """Pose blend shape bases."""
    
    v_template: Array
    """Canonical mesh verts."""
    
    shapedirs: Array
    exprdirs: Optional[Array] = None

    @staticmethod
    def load(pickle_path: Path) -> SmplxModel:
        # smpl_params: dict[str, onp.ndarray] = onp.load(npz_path, allow_pickle=True)

        with open(pickle_path, 'rb') as smpl_file:
            # smpl_params = Struct(**pickle.load(smpl_file, encoding='latin1'))
            smpl_params = pickle.load(smpl_file, encoding='latin1')

        # assert smpl_params["bs_style"].item() == b"lbs"
        # assert smpl_params["bs_type"].item() == b"lrotmin"
        valid_keys = ["f", "J_regressor", "kintree_table", "weights", "posedirs", "v_template", "shapedirs"]
        smpl_params["shapedirs"] = onp.array(smpl_params["shapedirs"], dtype=onp.float32)
        smpl_params["J_regressor"] = onp.array(smpl_params["J_regressor"], dtype=onp.float32)
        
        smpl_params = {k: _normalize_dtype(v) for k, v in smpl_params.items() if k in valid_keys}

        exprdirs = smpl_params.get("exprdirs", None)
        if exprdirs is not None:
            exprdirs = onp.array(exprdirs, dtype=onp.float32)
        return SmplxModel(
            faces=jnp.array(smpl_params["f"]),
            J_regressor=jnp.array(smpl_params["J_regressor"]),
            parent_indices=jnp.array(smpl_params["kintree_table"][0][1:] - 1),
            weights=jnp.array(smpl_params["weights"]),
            posedirs=jnp.array(smpl_params["posedirs"]),
            v_template=jnp.array(smpl_params["v_template"]),
            shapedirs=jnp.array(smpl_params["shapedirs"]),
            exprdirs=exprdirs,
        )
        

    def with_shape(
        self, 
        betas: Array,
        expression: Optional[Array] = None
    ) -> SmplxShaped:
        """Apply shape and expression parameters."""
        # Shape deformation
        num_betas = betas.shape[-1]
        assert num_betas <= 10
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        
        # Expression deformation
        verts_shaped = verts_with_shape  # default
        if expression is not None and self.exprdirs is not None:
            num_expressions = expression.shape[-1]
            verts_shaped = verts_with_shape + einsum(
                self.exprdirs[:, :, :num_expressions],
                expression,
                "verts xyz expr, ... expr -> ... verts xyz",
            )
        
        # Compute joint positions on the final shaped vertices
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_shaped,
            "joints verts, ... verts xyz -> ... joints xyz",
        )
        
        # Extract root and center everything
        root_offset = root_and_joints_pred[..., 0:1, :]
        return SmplxShaped(
            body_model=self,
            verts_zero=verts_shaped - root_offset,
            joints_zero=root_and_joints_pred[..., 1:, :] - root_offset,
            t_parent_joint=root_and_joints_pred[..., 1:, :]
            - root_and_joints_pred[..., self.parent_indices + 1, :],
        )


@jdc.pytree_dataclass
class SmplxShaped:
    """SMPL-X model with shape/expression applied."""
    
    body_model: SmplxModel
    verts_zero: Array
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Array
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Array
    """Position of each shaped body joint relative to its parent. Does not
    include root."""

    def with_pose_decomposed(
        self,
        T_world_root: Array | onp.ndarray,
        body_quats: Array | onp.ndarray,
    ) -> SmplxShapedAndPosed:
        """Pose our SMPL body model. Returns a set of joint and vertex outputs."""

        local_quats = broadcasting_cat(
            cast(list[jax.Array], [body_quats]),
            axis=0,
        )
        assert local_quats.shape[-2:] == (54, 4), local_quats.shape
        return self.with_pose(T_world_root, local_quats)
    
    def with_pose(
        self,
        T_world_root: Array,
        local_quats: Array,
    ) -> SmplxShapedAndPosed:
        """Apply pose with quaternion rotations."""
        
        # Forward kinematics.
        assert local_quats.shape == (54, 4), local_quats.shape
        # print(f"local_quats shape: {local_quats.shape}")
        
        parent_indices = self.body_model.parent_indices
        (num_joints,) = parent_indices.shape[-1:]
        num_active_joints, _ = local_quats.shape[-2:]
        assert local_quats.shape[-1] == 4
        assert num_active_joints <= num_joints
        assert self.t_parent_joint.shape[-2:] == (num_joints, 3)

        # Get relative transforms.
        Ts_parent_child = broadcasting_cat(
            [local_quats, self.t_parent_joint[..., :num_active_joints, :]], axis=-1
        )
        assert Ts_parent_child.shape[-2:] == (num_active_joints, 7)

        # Compute one joint at a time.
        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                parent_indices[i] == -1,
                T_world_root,
                Ts_world_joint[..., parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros_like(Ts_parent_child),
        )
        assert Ts_world_joint.shape[-2:] == (num_active_joints, 7)
        
        # print shapes
        # print(f"T_world_root shape: {T_world_root.shape}")
        # print(f"local_quats shape: {local_quats.shape}")
        # print(f"Ts_world_joint shape: {Ts_world_joint.shape}")

        return SmplxShapedAndPosed(
            shaped_model=self,
            T_world_root=T_world_root,  # type: ignore
            local_quats=local_quats,  # type: ignore
            Ts_world_joint=Ts_world_joint,
        )

    def with_pose_full(
        self,
        T_world_root: Array,
        body_pose: Array,
        left_hand_pose: Optional[Array] = None,
        right_hand_pose: Optional[Array] = None,
        jaw_pose: Optional[Array] = None,
    ) -> SmplxShapedAndPosed:
        """Apply full SMPL-X pose with optional hand/jaw articulation."""
        # Fast-path: if caller already supplies the *full* 54-joint quaternion stack
        #           (21 body + 3 jaw/eyes + 15 left-hand + 15 right-hand) we simply
        #           forward it directly.
        if body_pose.shape[-2] == 54:
            return self.with_pose(T_world_root, body_pose)

        # Otherwise we expect exactly the 21 body joints and will append the
        # jaw/eye and hand segments (filling missing ones with identity quats).

        batch_shape = body_pose.shape[:-2]
        identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])

        def _ensure_or_identity(pose: Optional[jax.Array], num_joints: int):
            if pose is not None:
                return pose
            # Produce an (…, num_joints, 4) array of identity quaternions that
            # matches the current batch shape.
            return jnp.broadcast_to(identity_quat, batch_shape + (num_joints, 4))



        # body_pose: 21 joints
        # 0: Pelvis (root)
        # 1: Left Hip
        # 2: Right Hip
        # 3: Spine1
        # 4: Left Knee
        # 5: Right Knee
        # 6: Spine2
        # 7: Left Ankle
        # 8: Right Ankle
        # 9: Spine3
        # 10: Left Foot
        # 11: Right Foot
        # 12: Neck
        # 13: Left Collar
        # 14: Right Collar
        # 15: Head
        # 16: Left Shoulder
        # 17: Right Shoulder
        # 18: Left Elbow
        # 19: Right Elbow
        # 20: Left Wrist
        # 21: Right Wrist
        
        # jaw_pose: 3 joints
        # 0: Jaw (open/close)
        # 1: left eye rotation
        # 2: right eye rotation
                
        # left_hand_pose: 15 joints
        # 0: Left Index1 (proximal phalanx)
        # 1: Left Index2 (middle phalanx)
        # 2: Left Index3 (distal phalanx)
        # 3: Left Middle1 (proximal phalanx)
        # 4: Left Middle2 (middle phalanx)
        # 5: Left Middle3 (distal phalanx)
        # 6: Left Pinky1 (proximal phalanx)
        # 7: Left Pinky2 (middle phalanx)
        # 8: Left Pinky3 (distal phalanx)
        # 9: Left Ring1 (proximal phalanx)
        # 10: Left Ring2 (middle phalanx)
        # 11: Left Ring3 (distal phalanx)
        # 12: Left Thumb1 (proximal phalanx)
        # 13: Left Thumb2 (middle phalanx)
        # 14: Left Thumb3 (distal phalanx)
        
        # right_hand_pose: 15 joints
        # 0: Right Index1 (proximal phalanx)
        # 1: Right Index2 (middle phalanx)
        # 2: Right Index3 (distal phalanx)
        # 3: Right Middle1 (proximal phalanx)
        # 4: Right Middle2 (middle phalanx)
        # 5: Right Middle3 (distal phalanx)
        # 6: Right Pinky1 (proximal phalanx)
        # 7: Right Pinky2 (middle phalanx)
        # 8: Right Pinky3 (distal phalanx)
        # 9: Right Ring1 (proximal phalanx)
        # 10: Right Ring2 (middle phalanx)
        # 11: Right Ring3 (distal phalanx)
        # 12: Right Thumb1 (proximal phalanx)
        # 13: Right Thumb2 (middle phalanx)
        # 14: Right Thumb3 (distal phalanx)
        
        pose_parts = [body_pose,
                      _ensure_or_identity(jaw_pose, 3),
                      _ensure_or_identity(left_hand_pose, 15),
                      _ensure_or_identity(right_hand_pose, 15)]

        local_quats = jnp.concatenate(pose_parts, axis=-2)
        return self.with_pose(T_world_root, local_quats)


@jdc.pytree_dataclass
class SmplxShapedAndPosed(SmplShapedAndPosed):
    """SMPL-X model with shape and pose applied."""
    
    shaped_model: SmplxShaped
    T_world_root: Array
    """Root coordinate frame."""

    local_quats: Array
    """Local joint orientations."""

    Ts_world_joint: Array
    """Absolute transform for each joint. Does not include the root."""
        
    def lbs(self) -> SmplxMesh:
        assert (
            self.local_quats.shape[0]
            == self.shaped_model.body_model.parent_indices.shape[0]
        ), "It looks like only a partial set of joint rotations was passed into `with_pose()`. We need all of them for LBS."

        # Linear blend skinning with a pose blend shape.
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (jaxlie.SO3(self.local_quats).as_matrix() - jnp.eye(3)).flatten(),
            "verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )
        # Build joints array with root prepend and broadcast batch dims
        root = jnp.zeros(self.shaped_model.joints_zero.shape[:-2] + (1, 3), dtype=self.shaped_model.joints_zero.dtype)
        joints = jnp.concatenate([root, self.shaped_model.joints_zero], axis=-2)  # (..., joints_p1, 3)

        # Vertex offsets w.r.t. each joint (homogeneous coordinates later)
        verts_minus_joints = verts_with_blend[..., :, None, :] - joints[..., None, :, :]  # (..., verts, joints_p1, 3)
        # Pad a 1 to make homogeneous coord dimension = 4
        pad_width = ((0, 0),) * (verts_minus_joints.ndim - 1) + ((0, 1),)
        verts_homo = jnp.pad(verts_minus_joints, pad_width, constant_values=1.0)

        verts_transformed = einsum(
            broadcasting_cat(
                [
                    jaxlie.SE3(self.T_world_root).as_matrix()[..., None, :3, :],  # (..., 1, 3, 4)
                    jaxlie.SE3(self.Ts_world_joint).as_matrix()[..., :3, :],       # (..., joints, 3, 4)
                ],
                axis=-3,  # concatenate along the joints axis
            ),
            self.shaped_model.body_model.weights,
            verts_homo,
            "... joints_p1 i j, ... verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )

        return SmplxMesh(
            posed_model=self,
            verts=verts_transformed,
            faces=self.shaped_model.body_model.faces,
        )


@jdc.pytree_dataclass
class SmplxMesh:
    """Final SMPL-X mesh output."""
    
    posed_model: SmplxShapedAndPosed
    verts: Array
    faces: Array

def broadcasting_cat(arrays: Sequence[jax.Array | onp.ndarray], axis: int) -> jax.Array:
    """Like jnp.concatenate, but broadcasts leading axes."""
    assert len(arrays) > 0
    output_dims = max(map(lambda t: len(t.shape), arrays))
    arrays = [t.reshape((1,) * (output_dims - len(t.shape)) + t.shape) for t in arrays]
    max_sizes = [max(t.shape[i] for t in arrays) for i in range(output_dims)]
    expanded_arrays = [
        jnp.broadcast_to(
            array,
            tuple(
                array.shape[i] if i == axis % len(array.shape) else max_size
                for i, max_size in enumerate(max_sizes)
            ),
        )
        for array in arrays
    ]
    return jnp.concatenate(expanded_arrays, axis=axis)


def _normalize_dtype(v: onp.ndarray) -> onp.ndarray:
    """Normalize datatypes; all arrays should be either int32 or float32."""
    if "int" in str(v.dtype):
        return v.astype(onp.int32)
    elif "float" in str(v.dtype):
        return v.astype(onp.float32)
    else:
        return v
    

if __name__ == "__main__":
    smpl_model = SmplModel.load(Path("./body_models/smpl/SMPL_NEUTRAL.pkl"))
    import pdb; pdb.set_trace()
    print(smpl_model)
