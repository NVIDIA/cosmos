# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch


def _recursive_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return type(x)(_recursive_to_numpy(v) for v in x)
    if isinstance(x, dict):
        return {k: _recursive_to_numpy(v) for k, v in x.items()}
    return x


def supports_numpy(arg_names, use_no_grad: bool = True):
    """Decorator to transparently support numpy inputs.

    - Converts the specified named args from numpy arrays to torch tensors on entry
    - Runs the wrapped function (optionally under no_grad)
    - Converts returns back to numpy iff the FIRST targeted arg was a numpy array
    """
    import functools
    import inspect

    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            first_is_numpy = False
            for idx, name in enumerate(arg_names):
                if name in bound.arguments:
                    val = bound.arguments[name]
                    # Handle direct ndarray
                    if isinstance(val, np.ndarray):
                        if idx == 0:
                            first_is_numpy = True
                        bound.arguments[name] = torch.from_numpy(val)
                        continue
                    # Handle list/tuple of ndarrays -> list/tuple of tensors
                    if isinstance(val, (list, tuple)):
                        saw_numpy_first_elem = False
                        converted = []
                        for i, el in enumerate(val):
                            if isinstance(el, np.ndarray):
                                if i == 0:
                                    saw_numpy_first_elem = True
                                converted.append(torch.from_numpy(el))
                            else:
                                converted.append(el)
                        if idx == 0 and saw_numpy_first_elem:
                            first_is_numpy = True
                        bound.arguments[name] = type(val)(converted)

            ctx = torch.no_grad() if use_no_grad else torch.enable_grad()
            with ctx:
                out = fn(*bound.args, **bound.kwargs)
            return _recursive_to_numpy(out) if first_is_numpy else out

        return wrapper

    return decorator


class Camera:
    """A class with a collection of common ops for camera transformations (Pytorch tensors).

    All poses are expected to have shape [...,3,4], where (...) indicates batch sizes of various ranks.
    The last two dimensions (of size (3,4)) correspond to the extrinsic matrix [R|t] in OpenCV format.

    Convention: cam_pose is always a world-to-camera transform (world2cam): x_cam = R @ x_world + t.
    This module operates on row-vector points with homogeneous coordinates on the right, so we apply
    transformations as: points_hom @ cam_pose^T.
    """

    @staticmethod
    @supports_numpy(["cam_pose"], use_no_grad=True)
    def _check_valid_pose(cam_pose: torch.Tensor | np.ndarray) -> None:
        """Checks whether the input tensor is a valid camera pose.

        Args:
            cam_pose (torch.Tensor [...,3,4]): Input camera pose in world2cam [R|t] (OpenCV) format.
        """
        assert cam_pose.shape[-2:] == (3, 4), "Camera pose is not of shape (3,4)."
        R = cam_pose[..., :3]
        # Compute determinant in float32 for numerical stability and allow dtype-dependent tolerance.
        det_R = torch.linalg.det(R.to(torch.float32))
        one = torch.tensor(1.0, dtype=torch.float32, device=cam_pose.device)
        if cam_pose.dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-6
        finite = bool(torch.isfinite(det_R).all())
        close = torch.allclose(det_R, one, rtol=rtol, atol=atol)
        assert finite and close, (
            f"Rotation component in camera pose is invalid (det != 1 within tol). "
            f"dtype={cam_pose.dtype}, rtol={rtol}, atol={atol}, det_mean={det_R.mean().item():.6f}"
        )

    @staticmethod
    @supports_numpy(["cam_pose"], use_no_grad=True)
    def invert_pose(cam_pose: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Invert a camera pose.

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): Input camera pose (world2cam [R|t]).

        Returns:
            cam_pose_inv (torch.Tensor/np.ndarray [...,3,4]): The inverted camera pose (cam2world [R|t]).
        """
        Camera._check_valid_pose(cam_pose)
        in_dtype = cam_pose.dtype if isinstance(cam_pose, torch.Tensor) else torch.float32
        R, t = cam_pose[..., :3], cam_pose[..., 3:]  # [...,3,3], [...,3,1]
        # Compute in float32 for numerical stability, cast back at the end
        R32 = R.to(torch.float32)  # [...,3,3]
        t32 = t.to(torch.float32)  # [...,3,1]
        # For rotation matrices, inverse equals transpose; prefer transpose for stability and speed
        R_inv32 = R32.transpose(-1, -2)  # [...,3,3]
        t_inv32 = -R_inv32 @ t32  # [...,3,1]
        cam_pose_inv32 = torch.cat([R_inv32, t_inv32], dim=-1)  # [...,3,4]
        return cam_pose_inv32.to(in_dtype)

    @staticmethod
    @supports_numpy(["cam_poses"], use_no_grad=True)
    def compose_poses(cam_poses: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
        """Compose a sequence of camera transformations together.

        pose_new = compose_poses([pose_1, pose_2, ... pose_N])
        pose_new(x) = pose_N o ... o pose_2 o pose_1(x)

        Args:
            cam_poses (list[torch.Tensor/np.ndarray [...,3,4]]): Sequence of rigid transforms [R|t].
                When used as camera extrinsics in this module, each pose is assumed to be world2cam.
                The composition follows the same row-vector convention: points_hom @ pose^T.
                List items may be numpy arrays; they will be converted to torch internally.

        Returns:
            cam_pose_new (torch.Tensor/np.ndarray [...,3,4]): The composed transformation [R|t].
        """
        cam_pose_new = cam_poses[0]
        Camera._check_valid_pose(cam_pose_new)
        out_dtype = cam_pose_new.dtype if isinstance(cam_pose_new, torch.Tensor) else torch.float32
        R_new, t_new = (
            cam_pose_new[..., :3].to(torch.float32),
            cam_pose_new[..., 3:].to(torch.float32),
        )  # [...,3,3], [...,3,1]
        for cam_pose in cam_poses[1:]:
            Camera._check_valid_pose(cam_pose)
            # pose_new(x) = pose o pose_new(x)
            R, t = cam_pose[..., :3].to(torch.float32), cam_pose[..., 3:].to(torch.float32)  # [...,3,3], [...,3,1]
            R_new = R @ R_new  # [...,3,3]
            t_new = R @ t_new + t  # [...,3,1]
        cam_pose_new32 = torch.cat([R_new, t_new], dim=-1)  # [...,3,4]
        return cam_pose_new32.to(out_dtype)

    @staticmethod
    @supports_numpy(["cam_pose", "cam_intr"], use_no_grad=True)
    def get_camera_rays(
        cam_pose: torch.Tensor | np.ndarray,
        cam_intr: torch.Tensor | np.ndarray,
        image_size: tuple[int, int],
    ) -> torch.Tensor | np.ndarray:
        """Get unit-norm camera rays in world coordinates for each pixel center.

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): Camera pose (world2cam [R|t]).
            cam_intr (torch.Tensor/np.ndarray [...,3,3]): Camera intrinsics.
            image_size (Tuple[int, int]): Image size (height, width).

        Returns:
            rays_world (torch.Tensor/np.ndarray [...,HW,3]): Unit direction rays from camera center through pixel centers, flattened over pixels.
        """
        H, W = image_size
        with torch.no_grad():
            # Compute image coordinate grid (in float32 for stability).
            y_range = torch.arange(H, dtype=torch.float32, device=cam_pose.device).add_(0.5)  # [H]
            x_range = torch.arange(W, dtype=torch.float32, device=cam_pose.device).add_(0.5)  # [W]
            y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W] each
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).view(-1, 2)  # [HW,2]
            xy_grid = xy_grid.repeat(*cam_pose.shape[:-2], 1, 1)  # [...,HW,2]
        # Pixel centers in camera coordinates at depth 1 (flattened HW)
        grid_camera = Camera.image2camera(Camera.to_homogeneous(xy_grid), cam_intr)  # [...,HW,3]
        # Transform sample points and center to world
        grid_world = Camera.camera2world(grid_camera, cam_pose)  # [...,HW,3]
        center_world = Camera.get_camera_center(cam_pose).unsqueeze(-2).expand_as(grid_world)  # [...,HW,3]
        rays_world = grid_world - center_world  # [...,HW,3]
        # Normalize to unit vectors
        eps = 1e-8
        if cam_pose.dtype in (torch.bfloat16, torch.float16):
            eps = 1e-2
        norms32 = rays_world.to(torch.float32).norm(dim=-1, keepdim=True).clamp_min(eps)  # [...,HW,1]
        rays_world = rays_world / norms32.to(rays_world.dtype)  # [...,HW,3]
        # Cast back to input dtype for consistency
        rays_world = rays_world.to(cam_pose.dtype)  # [...,HW,3]
        # Keep flattened shape [...,HW,3]
        return rays_world

    @staticmethod
    @supports_numpy(["cam_pose", "cam_intr"], use_no_grad=True)
    def get_plucker_rays(
        cam_pose: torch.Tensor | np.ndarray,
        cam_intr: torch.Tensor | np.ndarray,
        image_size: tuple[int, int],
    ) -> torch.Tensor | np.ndarray:
        """Get Plücker coordinates (moment, direction) for each pixel center.

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): Camera pose (world2cam [R|t]).
            cam_intr (torch.Tensor/np.ndarray [...,3,3]): Camera intrinsics.
            image_size (Tuple[int, int]): Image size (height, width).

        Returns:
            plucker (torch.Tensor/np.ndarray [...,HW,6]): Plücker coordinates [m | d], where
                d is a unit direction vector and m = o × d with o the camera center in world.
        """
        H, W = image_size
        rays_world = Camera.get_camera_rays(cam_pose, cam_intr, image_size)  # [...,HW,3]
        # Expand center to [...,HW,3]
        center_hw = Camera.get_camera_center(cam_pose).unsqueeze(-2).expand_as(rays_world)  # [...,HW,3]
        moment = torch.linalg.cross(center_hw, rays_world)  # [...,HW,3]
        plucker = torch.cat([moment, rays_world], dim=-1)  # [...,HW,6]
        return plucker

    @staticmethod
    @supports_numpy(["cam_pose"], use_no_grad=True)
    def get_relative_poses_wrt_frame0(
        cam_pose: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Compute poses relative to the first frame (index 0).

        All poses are world-to-camera [R|t] with shape [...,3,4]. The returned poses are expressed
        in the coordinate system of the first camera, so the first pose is identity [I|0]. For the
        i-th pose: pose_rel_i = compose(pose_i, inverse(pose_ref)).

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,V,3,4]): World-to-camera extrinsics per view.

        Returns:
            cam_pose_rel (torch.Tensor/np.ndarray [...,V,3,4]): Relative world-to-camera extrinsics in the first frame.
        """
        # supports_numpy handles numpy
        assert cam_pose.shape[-2:] == (3, 4), "cam_pose must have shape [..., V, 3, 4]."
        # Reference pose and its inverse
        pose_ref = cam_pose.select(dim=-3, index=0)  # [...,3,4]
        pose_ref_inv = Camera.invert_pose(pose_ref)  # [...,3,4]
        # Compose with broadcasting: pose_rel = pose ∘ pose_ref_inv
        cam_pose_rel = Camera.compose_poses([pose_ref_inv, cam_pose])  # [...,V,3,4]
        return cam_pose_rel

    @staticmethod
    @supports_numpy(["cam_pose"], use_no_grad=True)
    def get_camera_center(cam_pose: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Get the camera center in world coordinates for a given world2cam pose.

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): Camera pose (world2cam [R|t]).

        Returns:
            center_world (torch.Tensor/np.ndarray [...,3]): Camera center in world coordinates.
        """
        Camera._check_valid_pose(cam_pose)
        R, t = cam_pose[..., :3], cam_pose[..., 3:]  # [...,3,3], [...,3,1]
        center_world32 = (-R.to(torch.float32).transpose(-1, -2) @ t.to(torch.float32)).squeeze(-1)  # [...,3]
        return center_world32.to(R.dtype)

    @staticmethod
    @supports_numpy(["points"], use_no_grad=True)
    def to_homogeneous(points: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Get homogeneous coordinates of the input points.

        Args:
            points (torch.Tensor/np.ndarray [...,K]): Input coordinates.

        Returns:
            points_hom (torch.Tensor/np.ndarray [...,K+1]): Homogeneous coordinates.
        """
        # Compute homogeneous coordinate in float32 for stability, then cast back
        one32 = torch.ones_like(
            points[..., :1], dtype=torch.float32, device=(points.device if isinstance(points, torch.Tensor) else None)
        )  # [...,1]
        points_hom = torch.cat([points, one32.to(points.dtype)], dim=-1)  # [...,K+1]
        return points_hom

    @staticmethod
    @supports_numpy(["points", "cam_pose"], use_no_grad=True)
    def world2camera(
        points: torch.Tensor | np.ndarray, cam_pose: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Given the camera pose, transform input 3D points from world coordinates to camera coordinates.

        Args:
            points (torch.Tensor/np.ndarray [...,N,3]): Input 3D points.
            cam_pose (torch.Tensor/np.ndarray [...,3,4]/[3,4]): (Batched) camera pose (world2cam [R|t]).

        Returns:
            points_new (torch.Tensor/np.ndarray [...,N,3]): Transformed 3D points.
        """
        points_hom = Camera.to_homogeneous(points).to(torch.float32)  # [...,N,4]
        points_new32 = points_hom @ cam_pose.to(torch.float32).transpose(-1, -2)  # [...,N,3]
        return points_new32.to(points.dtype)  # [...,N,3]

    @staticmethod
    @supports_numpy(["points", "cam_pose"], use_no_grad=True)
    def camera2world(
        points: torch.Tensor | np.ndarray, cam_pose: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Given the camera pose, transform input 3D points from camera coordinates to world coordinates.

        Args:
            points (torch.Tensor/np.ndarray [...,N,3]): Input 3D points.
            cam_pose (torch.Tensor/np.ndarray [...,3,4]/[3,4]): (Batched) camera pose (world2cam [R|t]).

        Returns:
            points_new (torch.Tensor/np.ndarray [...,N,3]): Transformed 3D points.
        """
        points_hom = Camera.to_homogeneous(points).to(torch.float32)  # [...,N,4]
        pose_inv = Camera.invert_pose(cam_pose)  # [...,3,4]
        points_new32 = points_hom @ pose_inv.to(torch.float32).transpose(-1, -2)  # [...,N,3]
        # To reduce double-quantization error on low-precision dtypes (e.g., bf16 on CPU),
        # keep high precision on output for transform back to world space.
        if isinstance(points, torch.Tensor) and points.dtype in (torch.bfloat16, torch.float16):
            return points_new32  # [...,N,3]
        return points_new32.to(points.dtype)  # [...,N,3]

    @staticmethod
    @supports_numpy(["points", "cam_intr"], use_no_grad=True)
    def camera2image(
        points: torch.Tensor | np.ndarray, cam_intr: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Given the camera intrinsics, calibrate input 3D points from camera frame to image (pixel) frame.

        Args:
            points (torch.Tensor/np.ndarray [...,N,3]): Input 3D points.
            cam_intr (torch.Tensor/np.ndarray [...,3,3]/[3,3]): (Batched) camera intrinsic matrix.

        Returns:
            points_new (torch.Tensor/np.ndarray [...,N,3]): Transformed 3D points.
        """
        points32 = points.to(torch.float32)  # [...,N,3]
        points_new32 = points32 @ cam_intr.to(torch.float32).transpose(-1, -2)  # [...,N,3]
        return points_new32.to(points.dtype)  # [...,N,3]

    @staticmethod
    @supports_numpy(["points", "cam_intr"], use_no_grad=True)
    def image2camera(
        points: torch.Tensor | np.ndarray, cam_intr: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Given the camera intrinsics, calibrate input 3D points from image (pixel) frame to camera frame.

        Args:
            points (torch.Tensor/np.ndarray [...,N,3]): Input 3D points.
            cam_intr (torch.Tensor/np.ndarray [...,3,3]/[3,3]): (Batched) camera intrinsic matrix.

        Returns:
            points_new (torch.Tensor/np.ndarray [...,N,3]): Transformed 3D points.
        """
        K_inv32 = torch.linalg.inv(cam_intr.to(torch.float32))  # [...,3,3]
        points32 = points.to(torch.float32)  # [...,N,3]
        points_new32 = points32 @ K_inv32.transpose(-1, -2)  # [...,N,3]
        return points_new32.to(points.dtype)  # [...,N,3]

    @staticmethod
    @supports_numpy(["params"], use_no_grad=True)
    def intrinsic_params_to_matrices(params: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert (fx, fy, cx, cy) parameters to camera intrinsic matrix/matrices.

        Args:
            params (torch.Tensor/np.ndarray [...,4]): Intrinsic parameters (fx, fy, cx, cy).

        Returns:
            K (torch.Tensor/np.ndarray [...,3,3]): Camera intrinsic matrices.
        """
        assert params.shape[-1] == 4, "Intrinsic params must have shape (..., 4) for (fx, fy, cx, cy)."
        fx, fy, cx, cy = params.unbind(dim=-1)  # [...] each
        one = torch.ones_like(fx)  # [...]
        zero = torch.zeros_like(fx)  # [...]
        row0 = torch.stack([fx, zero, cx], dim=-1)  # [...,3]
        row1 = torch.stack([zero, fy, cy], dim=-1)  # [...,3]
        row2 = torch.stack([zero, zero, one], dim=-1)  # [...,3]
        K = torch.stack([row0, row1, row2], dim=-2)  # [...,3,3]
        return K

    @staticmethod
    @supports_numpy(["cam_intr"], use_no_grad=True)
    def intrinsic_matrices_to_params(
        cam_intr: torch.Tensor | np.ndarray, atol: float = 1e-6
    ) -> torch.Tensor | np.ndarray:
        """Extract (fx, fy, cx, cy) from camera intrinsic matrix/matrices.

        Args:
            cam_intr (torch.Tensor/np.ndarray [...,3,3]): Camera intrinsic matrices.
            atol (float): Tolerance when checking the bottom row against [0,0,1].

        Returns:
            params (torch.Tensor/np.ndarray [...,4]): Intrinsic parameters (fx, fy, cx, cy).
        """
        assert cam_intr.shape[-2:] == (3, 3), "Intrinsic matrix must have shape (..., 3, 3)."
        row32 = cam_intr[..., 2, :].to(torch.float32)
        target32 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=cam_intr.device)
        rtol = 1e-5
        atol_eff = atol
        if cam_intr.dtype in (torch.bfloat16, torch.float16):
            rtol = 1e-2
            atol_eff = max(atol, 1e-2)
        if not torch.allclose(row32, target32, rtol=rtol, atol=atol_eff):
            # Still proceed but warn via assertion message if strictness is desired.
            pass
        fx = cam_intr[..., 0, 0]  # [...]
        fy = cam_intr[..., 1, 1]  # [...]
        cx = cam_intr[..., 0, 2]  # [...]
        cy = cam_intr[..., 1, 2]  # [...]
        params = torch.stack([fx, fy, cx, cy], dim=-1)  # [...,4]
        return params

    @staticmethod
    @supports_numpy(["qxyzw_t"], use_no_grad=True)
    def extrinsic_params_to_matrices(qxyzw_t: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert (x,y,z,w, tx,ty,tz) to world2cam extrinsic matrix/matrices [R|t].

        Args:
            qxyzw_t (torch.Tensor/np.ndarray [...,7]): Quaternion (xyzw) and translation stacked.

        Returns:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): World-to-camera extrinsic [R|t].
        """
        assert qxyzw_t.shape[-1] == 7, "Input must have shape (..., 7) for (qx,qy,qz,qw,tx,ty,tz)."
        q = qxyzw_t[..., :4]  # [...,4]
        t = qxyzw_t[..., 4:7]  # [...,3]
        # Enforce unit quaternion
        Quaternion._check_valid_quaternion(q, require_normalized=True)
        R = Quaternion.to_rotation_matrix(q)  # [...,3,3]
        cam_pose = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [...,3,4]
        return cam_pose

    @staticmethod
    @supports_numpy(["cam_pose"], use_no_grad=True)
    def extrinsic_matrices_to_params(cam_pose: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert world2cam extrinsic matrix/matrices [R|t] to (x,y,z,w, tx,ty,tz).

        Args:
            cam_pose (torch.Tensor/np.ndarray [...,3,4]): World-to-camera extrinsic [R|t].

        Returns:
            qxyzw_t (torch.Tensor/np.ndarray [...,7]): Quaternion (xyzw) and translation stacked.
        """
        Camera._check_valid_pose(cam_pose)
        R = cam_pose[..., :3]  # [...,3,3]
        t = cam_pose[..., 3:].squeeze(-1)  # [...,3]
        q = Quaternion.from_rotation_matrix(R)  # [...,4]
        qxyzw_t = torch.cat([q, t], dim=-1)  # [...,7]
        return qxyzw_t


class Quaternion:
    """A collection of common quaternion operations (Pytorch tensors).

    Convention (STRICT): Quaternions are represented in (x, y, z, w) order (xyzw) and are unit-norm.
    The last dimension must be size 4.
    """

    @staticmethod
    @supports_numpy(["q"], use_no_grad=True)
    def _check_valid_quaternion(
        q: torch.Tensor | np.ndarray, require_normalized: bool = True, atol: float = 1e-5
    ) -> torch.Tensor | np.ndarray:
        """Checks whether the input tensor is a valid quaternion.

        Args:
            q (torch.Tensor [...,4]): Input quaternion(s) in (x, y, z, w) order.
            require_normalized (bool): If True, assert unit-norm within atol. Defaults to True.
            atol (float): Absolute tolerance for the unit-norm check.
        """
        assert q.shape[-1] == 4, "Quaternion is not of shape (..., 4)."
        if require_normalized:
            norms32 = q.to(torch.float32).norm(dim=-1)  # [...]
            ones32 = torch.ones_like(norms32)  # [...]
            tol = max(atol, 1e-2) if q.dtype in (torch.bfloat16, torch.float16) else atol
            assert torch.allclose(norms32, ones32, atol=tol), "Quaternion must be unit length."
        return q

    @staticmethod
    @supports_numpy(["q"], use_no_grad=True)
    def normalize(q: torch.Tensor | np.ndarray, eps: float = 1e-8) -> torch.Tensor | np.ndarray:
        """Normalize quaternion(s) to unit length.

        Args:
            q (torch.Tensor [...,4]): Input quaternion(s).
            eps (float): Small epsilon to avoid division by zero.

        Returns:
            q_norm (torch.Tensor [...,4]): Unit quaternions.
        """
        # Allow non-normalized input here, since this function normalizes
        Quaternion._check_valid_quaternion(q, require_normalized=False)
        eps_eff = eps
        if q.dtype in (torch.bfloat16, torch.float16):
            eps_eff = max(eps, 1e-2)
        norm32 = q.to(torch.float32).norm(dim=-1, keepdim=True).clamp_min(eps_eff)  # [...,1]
        out32 = q.to(torch.float32) / norm32  # [...,4]
        out = out32.to(q.dtype)  # [...,4]
        return out

    @staticmethod
    @supports_numpy(["q"], use_no_grad=True)
    def to_rotation_matrix(q: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert quaternion(s) to rotation matrix/matrices.

        Args:
            q (torch.Tensor [...,4]): Quaternion(s) (x, y, z, w).

        Returns:
            R (torch.Tensor [...,3,3]): Rotation matrix/matrices.
        """
        # Enforce unit quaternions for rotations
        Quaternion._check_valid_quaternion(q, require_normalized=True)
        q32 = q.to(torch.float32)  # [...,4]
        qx, qy, qz, qw = q32.unbind(dim=-1)  # [...] each
        two = torch.tensor(2.0, dtype=torch.float32, device=q32.device)

        r00 = 1 - two * (qy * qy + qz * qz)  # [...]
        r01 = two * (qx * qy - qz * qw)  # [...]
        r02 = two * (qx * qz + qy * qw)  # [...]
        r10 = two * (qx * qy + qz * qw)  # [...]
        r11 = 1 - two * (qx * qx + qz * qz)  # [...]
        r12 = two * (qy * qz - qx * qw)  # [...]
        r20 = two * (qx * qz - qy * qw)  # [...]
        r21 = two * (qx * qw + qy * qz)  # [...]
        r22 = 1 - two * (qx * qx + qy * qy)  # [...]

        R32 = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),  # [...,3]
                torch.stack([r10, r11, r12], dim=-1),  # [...,3]
                torch.stack([r20, r21, r22], dim=-1),  # [...,3]
            ],
            dim=-2,
        )  # [...,3,3]
        return R32.to(q.dtype)

    @staticmethod
    @supports_numpy(["R"], use_no_grad=True)
    def from_rotation_matrix(R: torch.Tensor | np.ndarray, eps: float = 1e-8) -> torch.Tensor | np.ndarray:
        """Convert rotation matrix/matrices to quaternion(s).

        Args:
            R (torch.Tensor [...,3,3]): Rotation matrix/matrices.
            eps (float): Numerical stability epsilon.

        Returns:
            q (torch.Tensor [...,4]): Quaternion(s) in (x, y, z, w) order.
        """
        assert R.shape[-2:] == (3, 3), "Rotation matrix is not of shape (..., 3, 3)."
        R32 = R.to(torch.float32)  # [...,3,3]
        m00 = R32[..., 0, 0]  # [...]
        m11 = R32[..., 1, 1]  # [...]
        m22 = R32[..., 2, 2]  # [...]
        trace = m00 + m11 + m22  # [...]

        q32 = torch.empty(*R32.shape[:-2], 4, dtype=torch.float32, device=R32.device)  # [...,4]

        cond0 = trace > 0  # [...]
        eps_eff = max(eps, 1e-6)
        s0 = torch.sqrt(trace + 1.0 + eps_eff) * 2.0  # [...]
        qw0 = 0.25 * s0  # [...]
        qx0 = (R[..., 2, 1] - R[..., 1, 2]) / s0  # [...]
        qy0 = (R[..., 0, 2] - R[..., 2, 0]) / s0  # [...]
        qz0 = (R[..., 1, 0] - R[..., 0, 1]) / s0  # [...]

        cond1 = (~cond0) & (m00 > m11) & (m00 > m22)  # [...]
        s1 = torch.sqrt(1.0 + m00 - m11 - m22 + eps_eff) * 2.0  # [...]
        qw1 = (R[..., 2, 1] - R[..., 1, 2]) / s1  # [...]
        qx1 = 0.25 * s1  # [...]
        qy1 = (R[..., 0, 1] + R[..., 1, 0]) / s1  # [...]
        qz1 = (R[..., 0, 2] + R[..., 2, 0]) / s1  # [...]

        cond2 = (~cond0) & (~cond1) & (m11 > m22)  # [...]
        s2 = torch.sqrt(1.0 + m11 - m00 - m22 + eps_eff) * 2.0  # [...]
        qw2 = (R[..., 0, 2] - R[..., 2, 0]) / s2  # [...]
        qx2 = (R[..., 0, 1] + R[..., 1, 0]) / s2  # [...]
        qy2 = 0.25 * s2  # [...]
        qz2 = (R[..., 1, 2] + R[..., 2, 1]) / s2  # [...]

        cond3 = (~cond0) & (~cond1) & (~cond2)  # [...]
        s3 = torch.sqrt(1.0 + m22 - m00 - m11 + eps_eff) * 2.0  # [...]
        qw3 = (R[..., 1, 0] - R[..., 0, 1]) / s3  # [...]
        qx3 = (R[..., 0, 2] + R[..., 2, 0]) / s3  # [...]
        qy3 = (R[..., 1, 2] + R[..., 2, 1]) / s3  # [...]
        qz3 = 0.25 * s3  # [...]

        qw = torch.where(cond0, qw0, torch.where(cond1, qw1, torch.where(cond2, qw2, qw3)))  # [...]
        qx = torch.where(cond0, qx0, torch.where(cond1, qx1, torch.where(cond2, qx2, qx3)))  # [...]
        qy = torch.where(cond0, qy0, torch.where(cond1, qy1, torch.where(cond2, qy2, qy3)))  # [...]
        qz = torch.where(cond0, qz0, torch.where(cond1, qz1, torch.where(cond2, qz2, qz3)))  # [...]

        q32[..., 0] = qx
        q32[..., 1] = qy
        q32[..., 2] = qz
        q32[..., 3] = qw

        q_out = Quaternion.normalize(q32).to(R.dtype)  # [...,4]
        return q_out

    @staticmethod
    @supports_numpy(["q"], use_no_grad=True)
    def invert(q: torch.Tensor | np.ndarray, eps: float = 1e-8) -> torch.Tensor | np.ndarray:
        """Inverse of quaternion(s). Equivalent to conjugate of quaternion.

        For unit quaternions, the inverse equals the conjugate. For non-unit quaternions,
        q^{-1} = conjugate(q) / ||q||^2.

        Args:
            q (torch.Tensor [...,4]): Input quaternion(s).
            eps (float): Small epsilon to avoid division by zero.

        Returns:
            q_inv (torch.Tensor [...,4]): Inverted quaternion(s).
        """
        # Enforce unit quaternions; inverse equals conjugate
        Quaternion._check_valid_quaternion(q, require_normalized=True)
        qx, qy, qz, qw = q.unbind(dim=-1)  # [...] each
        return torch.stack([-qx, -qy, -qz, qw], dim=-1)  # [...,4]

    @staticmethod
    @supports_numpy(["q1", "q2"], use_no_grad=True)
    def multiply(q1: torch.Tensor | np.ndarray, q2: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Hamilton product of two quaternion sets.

        Args:
            q1 (torch.Tensor [...,4]): Left quaternion(s) in (x, y, z, w) order.
            q2 (torch.Tensor [...,4]): Right quaternion(s) in (x, y, z, w) order.

        Returns:
            q (torch.Tensor [...,4]): Product quaternion(s) q = q1 ⊗ q2 in (x, y, z, w) order.
        """
        # Enforce unit inputs
        Quaternion._check_valid_quaternion(q1, require_normalized=True)
        Quaternion._check_valid_quaternion(q2, require_normalized=True)
        q1x, q1y, q1z, q1w = q1.unbind(dim=-1)  # [...] each
        q2x, q2y, q2z, q2w = q2.unbind(dim=-1)  # [...] each
        qx = q1w * q2x + q2w * q1x + q1y * q2z - q1z * q2y  # [...]
        qy = q1w * q2y + q2w * q1y + q1z * q2x - q1x * q2z  # [...]
        qz = q1w * q2z + q2w * q1z + q1x * q2y - q1y * q2x  # [...]
        qw = q1w * q2w - (q1x * q2x + q1y * q2y + q1z * q2z)  # [...]
        q = torch.stack([qx, qy, qz, qw], dim=-1)  # [...,4]
        # Re-normalize to counteract numerical drift and enforce constraint
        return Quaternion.normalize(q)  # [...,4]
