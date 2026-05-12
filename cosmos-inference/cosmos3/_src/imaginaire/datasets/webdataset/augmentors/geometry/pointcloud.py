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

"""Point cloud augmentors for webdataset."""

from typing import Optional

import torch
from einops import rearrange

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.modules.camera import Camera


class DepthToPointcloud(Augmentor):
    """Converts depth images to point clouds using camera intrinsics.

    This augmentor takes a depth image and camera intrinsics to generate a point cloud.
    The depth image should be in meters and the intrinsics should be a 3x3 matrix.

    Args:
        to_world_coords (bool): If True, uses the first frame as the coordinate frame for video sequences
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        """Initialize the depth to point cloud converter.

        Args:
            input_keys: List of input keys (typically ['depth', 'intrinsics', 'world_to_cam'])
            output_keys: List of output keys (typically ['points'])
            args: Additional arguments including:
                - to_world_coords (bool): Whether to use first frame as coordinate frame
        """
        assert "depth" in input_keys, "Depth image is required for point cloud conversion"
        assert "intrinsics" in input_keys, "Intrinsics are required for point cloud conversion"
        assert "world_to_cam" in input_keys or not self.to_world_coords, (
            "World to camera matrix is required for point cloud conversion"
        )
        super().__init__(input_keys, output_keys, args)
        self.to_world_coords = args.get("to_world_coords", False) if args else False

    def __call__(self, data_dict: dict) -> dict:
        """Convert depth image to point cloud.

        Args:
            data_dict: Input data dictionary containing depth image and camera intrinsics

        Returns:
            data_dict: Output data dictionary with point cloud
        """
        # Get depth image and intrinsics
        depth = data_dict[self.input_keys[0]]  # [T,H,W] or [H,W]
        intrinsics = data_dict[self.input_keys[1]]  # [T,3,3] or [3,3]

        # Check if we're dealing with video sequences (temporal dimension)
        if depth.dim() == 3 and intrinsics.dim() == 3:
            # Video sequence: T x H x W and T x 3 x 3
            T, H, W = depth.shape

            # Create pixel coordinates (same for all frames)
            y, x = torch.meshgrid(
                torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij"
            )
            pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H,W,3]
            pixels_hw3 = pixels.reshape(-1, 3)  # [H*W,3]

            # Back-project to camera space using Camera.image2camera
            pixels_batched = pixels_hw3.unsqueeze(0).expand(T, -1, -1)  # [T,H*W,3]
            points_cam = Camera.image2camera(pixels_batched, intrinsics)  # [T,H*W,3]
            depth_flat = depth.reshape(T, -1)  # [T,H*W]
            points_cam = points_cam * depth_flat.unsqueeze(-1)  # [T,H*W,3]

            # Transform to first frame coordinate system if requested
            if self.to_world_coords:
                world_to_cam = data_dict[self.input_keys[2]]  # [T,4,4]
                w2c = world_to_cam[:, :3, :]  # [T,3,4]
                # relative pose from cam_t to cam_0: rel = w2c_0 ∘ c2w_t
                w2c0 = w2c[0]  # [3,4]
                c2w = Camera.invert_pose(w2c)  # [T,3,4]
                w2c0_exp = w2c0.unsqueeze(0).expand_as(c2w)  # [T,3,4]
                rel = Camera.compose_poses([w2c0_exp, c2w])  # [T,3,4]
                points = Camera.world2camera(points_cam, rel)  # [T,H*W,3]
            else:
                points = points_cam  # [T,H*W,3]

            # Reshape to T x 3 x H x W
            points = rearrange(points, "t (h w) c -> c t h w", h=H, w=W, c=3)  # [3,T,H,W]

        else:
            # Single frame: H x W and 3 x 3
            H, W = depth.shape[-2:]

            # Create pixel coordinates
            y, x = torch.meshgrid(
                torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij"
            )

            # Create homogeneous coordinates and convert to float
            pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H,W,3]
            pixels_hw3 = pixels.reshape(-1, 3)  # [H*W,3]
            depth_flat = depth.reshape(-1)  # [H*W]

            # Back-project to camera space
            points_cam = Camera.image2camera(pixels_hw3, intrinsics)  # [H*W,3]
            points_cam = points_cam * depth_flat.unsqueeze(-1)  # [H*W,3]

            # For single frame, just use camera coordinates or transform to world coords as before
            if self.to_world_coords:
                world_to_cam = data_dict[self.input_keys[2]]  # [4,4]
                w2c = world_to_cam[:3, :]  # [3,4]
                points = Camera.camera2world(points_cam, w2c)  # [H*W,3]
            else:
                points = points_cam  # [H*W,3]

            # Reshape to 3 x H x W
            points = rearrange(points, "(h w) c -> c h w", h=H, w=W, c=3)  # [3,H,W]

        # Store in output dictionary
        data_dict[self.output_keys[0]] = points

        return data_dict


class PointcloudRescale(Augmentor):
    """Rescales point clouds to have a mean distance of 1 from the origin.

    This augmentor takes a point cloud and rescales it so that the mean distance
    of all points from the origin is 1. It also adjusts the world-to-camera
    transformation matrix accordingly.

    Args:
        input_keys: List of input keys (typically ['points', 'world_to_cam'])
        output_keys: List of output keys (typically ['points', 'world_to_cam'])
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        mask_key: Optional[str] = None,
        args: Optional[dict] = None,
    ) -> None:
        """Initialize the point cloud rescaler.

        Args:
            input_keys: List of input keys (typically ['points', 'world_to_cam'])
            output_keys: List of output keys (typically ['points', 'world_to_cam'])
            args: Additional arguments (not used in this augmentor)
        """
        assert "points" in input_keys, "Points are required for rescaling"
        assert "world_to_cam" in input_keys, "World to camera matrix is required for rescaling"
        super().__init__(input_keys, output_keys, args)
        self.mask_key = mask_key

    def __call__(self, data_dict: dict) -> dict:
        """Rescale point cloud and adjust world-to-camera transformation.

        This augmentor computes the average Euclidean distance of all 3D points to the origin
        and uses this scale to normalize both the camera translations and point cloud.

        Args:
            data_dict: Input data dictionary containing points and world_to_cam

        Returns:
            data_dict: Output data dictionary with rescaled points and adjusted world_to_cam
        """
        # Get points and world_to_cam
        points = data_dict[self.input_keys[0]]  # [3,T,H,W] or [3,H,W]
        world_to_cam = data_dict[self.input_keys[1]]  # [T,4,4] or [4,4]

        # Check if we're dealing with video sequences (temporal dimension)
        if points.dim() == 4 and world_to_cam.dim() == 3:
            # Video sequence: 3 x T x H x W and T x 4 x 4
            T = world_to_cam.shape[0]

            # Reshape points to T x N x 3 for easier computation
            points_flat = points.permute(1, 0, 2, 3).reshape(T, 3, -1).transpose(1, 2)  # [T,N,3]

            # Compute average Euclidean distance to origin across all frames
            if self.mask_key is not None:
                # Get mask and reshape to match points
                mask = data_dict[self.mask_key]  # [T,H,W]
                mask_flat = mask.reshape(T, -1)  # [T,N]

                # Only compute average over valid points across all frames
                # Compute squared distances for all frames at once
                squared_distances = torch.sum(points_flat**2, dim=2)  # [T,N]

                # Apply mask and compute mean across all frames
                valid_distances = torch.sqrt(squared_distances[mask_flat])  # [N_valid]
                avg_dist = valid_distances.mean()  # scalar
            else:
                # Compute average Euclidean distance to origin for all points across all frames
                avg_dist = torch.sqrt(torch.sum(points_flat**2, dim=2)).mean()  # scalar

            # Compute scale factor to achieve average distance of 1 across all frames
            scale = 1.0 / avg_dist  # scalar

            # Rescale points for all frames at once
            points_scaled = points * scale  # [3,T,H,W]

            # Adjust world_to_cam matrix for all frames at once
            # We need to scale the translation component by the same factor
            world_to_cam_scaled = world_to_cam.clone()
            world_to_cam_scaled[:, :3, 3] *= scale  # [T,4,4]

            # Scale depth for all frames at once
            depth = data_dict[self.input_keys[2]]  # [T,H,W]
            depth_scaled = depth * scale  # [T,H,W]
        else:
            # Single frame: 3 x H x W and 4 x 4
            # Reshape points to N x 3 for easier computation
            points_flat = points.reshape(3, -1).T  # [N,3]

            # Compute average Euclidean distance to origin
            if self.mask_key is not None:
                # Get mask and reshape to match points
                mask = data_dict[self.mask_key]  # [H,W]
                mask_flat = mask.reshape(-1)  # [N]

                # Only compute average over valid points
                valid_points = points_flat[mask_flat]  # [N_valid,3]
                # Compute average Euclidean distance to origin
                avg_dist = torch.sqrt(torch.sum(valid_points**2, dim=1)).mean()  # scalar
            else:
                # Compute average Euclidean distance to origin for all points
                avg_dist = torch.sqrt(torch.sum(points_flat**2, dim=1)).mean()  # scalar

            # Compute scale factor to achieve average distance of 1
            scale = 1.0 / avg_dist  # scalar

            # Rescale points
            points_scaled = points * scale  # [3,H,W]

            # Adjust world_to_cam matrix
            # We need to scale the translation component by the same factor
            world_to_cam_scaled = world_to_cam.clone()
            world_to_cam_scaled[:3, 3] *= scale  # [4,4]

            # Scale depth
            depth = data_dict[self.input_keys[2]]  # [H,W]
            depth_scaled = depth * scale  # [H,W]

        # Store in output dictionary
        data_dict[self.output_keys[0]] = points_scaled
        data_dict[self.output_keys[1]] = world_to_cam_scaled
        data_dict[self.output_keys[2]] = depth_scaled
        return data_dict


class PointcloudMaskFill(Augmentor):
    """Fills point cloud values with 0 when point cloud mask is False.

    This augmentor takes a point cloud and a point cloud mask, and sets point cloud values to 0
    wherever the mask is False. This is useful for cleaning up point clouds by
    removing invalid or unreliable point measurements.

    Args:
        input_keys: List of input keys (typically ['points', 'pcd_mask'])
        output_keys: List of output keys (typically ['points'])
    """

    def __init__(
        self, input_keys: list, output_keys: Optional[list] = None, fill_value: float = 0.0, args: Optional[dict] = None
    ) -> None:
        """Initialize the point cloud mask filler.

        Args:
            input_keys: List of input keys (typically ['points', 'pcd_mask'])
            output_keys: List of output keys (typically ['points'])
            args: Additional arguments (not used in this augmentor)
        """
        super().__init__(input_keys, output_keys, args)
        self.fill_value = fill_value

    def __call__(self, data_dict: dict) -> dict:
        """Fill point cloud values with 0 where point cloud mask is False.

        Args:
            data_dict: Input data dictionary containing point cloud and point cloud mask

        Returns:
            data_dict: Output data dictionary with masked point cloud
        """
        # Get point cloud and point cloud mask
        points = data_dict[self.input_keys[0]]  # [3,T,H,W] or [3,H,W]
        depth_mask = data_dict[self.input_keys[1]]  # [T,H,W] or [H,W]

        # Check if we're dealing with video sequences (temporal dimension)
        if points.dim() == 4 and depth_mask.dim() == 3:
            # Video sequence: 3 x T x H x W and T x H x W
            # Create a copy of the point cloud
            points_filled = points.clone()  # [3,T,H,W]

            # Expand mask to match points dimensions: 3 x T x H x W
            mask_expanded = depth_mask.unsqueeze(0).expand(3, -1, -1, -1)  # [3,T,H,W]

            # Set point cloud values to fill_value where mask is False for all channels at once
            points_filled[~mask_expanded] = self.fill_value

        else:
            # Single frame: 3 x H x W and H x W
            # Create a copy of the point cloud
            points_filled = points.clone()  # [3,H,W]

            # Expand mask to match points dimensions: 3 x H x W
            mask_expanded = depth_mask.unsqueeze(0).expand(3, -1, -1)  # [3,H,W]

            # Set point cloud values to fill_value where mask is False for all channels at once
            points_filled[~mask_expanded] = self.fill_value

        # Store in output dictionary
        data_dict[self.output_keys[0]] = points_filled

        return data_dict


def verify_backprojection(data_dict: dict, scale: float) -> bool:
    """Verify that backprojection of rescaled depth and camera poses matches rescaled point cloud.

    This function checks if the backprojection of the rescaled depth image using
    the rescaled camera poses produces the same point cloud as the rescaled point cloud.

    Args:
        data_dict: Dictionary containing:
            - points_scaled: Rescaled point cloud (3 x H x W)
            - depth_scaled: Rescaled depth image (H x W)
            - world_to_cam_scaled: Rescaled world to camera matrix (4 x 4)
            - intrinsics: Camera intrinsics matrix (3 x 3)
        scale: The scale factor used for rescaling

    Returns:
        bool: True if backprojection matches rescaled point cloud within tolerance
    """
    # Get required data
    points_scaled = data_dict["points"]  # [3,H,W]
    depth_scaled = data_dict["depth"]  # [H,W]
    world_to_cam_scaled = data_dict["world_to_cam"]  # [4,4]
    intrinsics = data_dict["intrinsics"]  # [3,3]

    # Get image dimensions
    H, W = depth_scaled.shape[-2:]

    # Create pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=depth_scaled.device), torch.arange(W, device=depth_scaled.device), indexing="ij"
    )

    # Create homogeneous coordinates
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H,W,3]

    # Reshape for batch processing
    pixels = pixels.reshape(-1, 3)  # [H*W,3]
    depth_flat = depth_scaled.reshape(-1)  # [H*W]

    # Get inverse of intrinsics
    intrinsics_inv = torch.inverse(intrinsics)  # [3,3]

    # Back-project to camera space
    points_cam = (intrinsics_inv @ pixels.T).T  # [H*W,3]
    points_cam = points_cam * depth_flat.unsqueeze(-1)  # [H*W,3]

    # Convert to world coordinates
    cam_to_world = torch.inverse(world_to_cam_scaled)  # [4,4]
    points_cam_h = torch.cat([points_cam, torch.ones_like(points_cam[:, :1])], dim=-1)  # [H*W,4]
    points_world_h = (cam_to_world @ points_cam_h.T).T  # [H*W,4]
    points_world = points_world_h[:, :3]  # [H*W,3]

    # Reshape back to image dimensions
    points_world = points_world.reshape(H, W, 3)  # [H,W,3]
    points_world = points_world.permute(2, 0, 1)  # [3,H,W]

    # Compare with rescaled point cloud
    # Use a small tolerance for floating point comparison
    tolerance = 1e-6
    is_close = torch.allclose(points_world, points_scaled, rtol=tolerance, atol=tolerance)

    return is_close
