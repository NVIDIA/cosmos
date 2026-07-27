"""
Video base64 caching to avoid reprocessing videos.
Significantly reduces memory usage and processing time.

Supports S3 fallback: local cache -> S3 cache -> generate new cache
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..smp import get_logger

logger = get_logger('VideoCache')


class VideoCache:
    """Cache for base64-encoded videos with processing parameters.

    Cache hierarchy:
    1. Local cache (fast)
    2. S3 cache (fallback)
    3. Generate new cache (save to local + upload to S3)
    """

    # S3 configuration for cache bucket
    S3_PFOFILE = 'team-cosmos'
    S3_BUCKET = 'cosmos_understanding'
    S3_PREFIX = 'video_cache'
    S3_ENDPOINT = 'https://pdx.s8k.io'

    def __init__(self, cache_dir=None, enable_s3=True):
        if cache_dir is None:
            from ..smp import LMUDataRoot
            cache_dir = os.path.join(LMUDataRoot(), 'video_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_s3 = enable_s3
        self._s3 = None  # Lazy initialization
        logger.info(f"Video cache directory: {self.cache_dir}")
        if self.enable_s3:
            logger.info(f"S3 cache enabled: s3://{self.S3_BUCKET}/{self.S3_PREFIX}")

    def _get_cache_key(self, video_path, **kwargs):
        """
        Generate cache key from video path and processing parameters.

        Args:
            video_path: Path to video file
            **kwargs: Processing parameters (fps, nframes, total_pixels, etc.)

        Returns:
            str: Cache key (hash)
        """
        # Create deterministic string from video file name + parameters
        video_name = os.path.basename(video_path)

        # Get file modification time as part of key (detect if video changed)
        try:
            file_size = os.path.getsize(video_path)
        except Exception:
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Sort kwargs for deterministic hashing
        params_str = json.dumps(kwargs, sort_keys=True)

        # Create hash
        key_string = f"{video_name}|{file_size}|{params_str}"
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()

        return cache_key

    def _get_cache_path(self, cache_key):
        """Get filesystem path for cache entry."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.cache"

    def _get_s3_path(self, cache_key):
        """Get S3 path for cache entry."""
        subdir = cache_key[:2]
        return f"{self.S3_BUCKET}/{self.S3_PREFIX}/{subdir}/{cache_key}.cache"

    def _get_s3_filesystem(self):
        """Get or create S3 filesystem connection (lazy initialization)."""
        if not self.enable_s3:
            return None

        if self._s3 is None:
            try:
                from s3fs import S3FileSystem

                # Try using environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                # Fallback to profile if env vars not available.
                # Bounded timeouts so a hung TCP connection can't block the worker forever.
                self._s3 = S3FileSystem(
                    anon=False,
                    profile=self.S3_PFOFILE,
                    client_kwargs={'endpoint_url': self.S3_ENDPOINT},
                    config_kwargs={
                        'connect_timeout': 10,
                        'read_timeout': 60,
                        'retries': {'max_attempts': 3},
                    },
                )
                logger.info("Connected to S3 using profile: team-cosmos")
            except ImportError:
                logger.warning("s3fs not installed, S3 cache disabled. Install with: pip install s3fs")
                self.enable_s3 = False
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize S3 connection: {e}")
                self.enable_s3 = False
                return None

        return self._s3

    def _download_from_s3(self, cache_key, local_path):
        """Download cache file from S3 to local path.

        Returns:
            bool: True if successful, False otherwise
        """
        s3 = self._get_s3_filesystem()
        if s3 is None:
            return False

        s3_path = self._get_s3_path(cache_key)
        temp_path = local_path.with_suffix('.tmp')

        try:
            if not s3.exists(s3_path):
                return False

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download atomically (temp file + rename)
            s3.get(s3_path, str(temp_path))
            temp_path.replace(local_path)

            logger.info(f"Cache downloaded from S3: {cache_key[:8]}...")
            return True

        except Exception as e:
            logger.warning(f"Failed to download from S3 ({cache_key[:8]}...): {e}")
            # Clean up temp file if exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False

    def _upload_to_s3(self, cache_key, local_path):
        """Upload cache file from local to S3.

        Returns:
            bool: True if successful, False otherwise
        """
        s3 = self._get_s3_filesystem()
        if s3 is None:
            return False

        s3_path = self._get_s3_path(cache_key)
        try:
            # Upload to S3
            s3.put(str(local_path), s3_path)
            logger.debug(f"Cache uploaded to S3: {cache_key} {s3_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to upload to S3 ({cache_key}): {e}")
            return False

    def get(self, video_path, **kwargs):
        """
        Get cached base64 video if exists.

        Cache hierarchy:
        1. Check local cache first (fast)
        2. If not found locally, try downloading from S3
        3. If not in S3 either, return None

        Args:
            video_path: Path to video file
            **kwargs: Processing parameters

        Returns:
            tuple: (video_url, video_kwargs) or (None, None) if not cached
        """
        cache_key = self._get_cache_key(video_path, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        # 1. Check local cache first
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)

                logger.debug(f"Cache HIT (local): {os.path.basename(video_path)}")
                return cache_data['video_url'], cache_data['video_kwargs']

            except Exception as e:
                logger.warning(f"Failed to load local cache for {cache_path}: {e}")
                # Remove corrupted cache file
                try:
                    cache_path.unlink()
                except Exception:
                    pass
                # Continue to try S3

        # 2. Try downloading from S3
        if self.enable_s3:
            if self._download_from_s3(cache_key, cache_path):
                # Successfully downloaded, now load it
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)

                    logger.debug(f"Cache HIT (S3): {os.path.basename(video_path)}")
                    return cache_data['video_url'], cache_data['video_kwargs']

                except Exception as e:
                    logger.warning(f"Failed to load cache after S3 download: {e}")
                    # Remove corrupted file
                    try:
                        cache_path.unlink()
                    except Exception:
                        pass

        # 3. Not found in local or S3
        return None, None

    def set(self, video_path, video_url, video_kwargs, **processing_kwargs):
        """
        Save base64 video to cache.

        Saves to:
        1. Local cache (always)
        2. S3 cache (if enabled, best-effort)

        Args:
            video_path: Path to video file
            video_url: Base64-encoded video data URL
            video_kwargs: Video processing metadata
            **processing_kwargs: Processing parameters used
        """
        cache_key = self._get_cache_key(video_path, **processing_kwargs)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Ensure parent directory exists (race condition safe)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                'video_url': video_url,
                'video_kwargs': video_kwargs,
                'video_path': video_path,
                'processing_kwargs': processing_kwargs
            }
            # Write temp file on the same filesystem as cache_path so that
            # the final replace() is an atomic rename (no cross-device copy).
            fd, tmp_str = tempfile.mkstemp(dir=str(cache_path.parent), suffix='.tmp')
            temp_path = Path(tmp_str)
            try:
                with open(fd, 'w') as f:
                    json.dump(cache_data, f)
                logger.debug(f"Cache MISS: Saved {os.path.basename(video_path)}")
                # Upload to S3 (best-effort, don't fail if it doesn't work)
                if self.enable_s3:
                    print(f"Uploading to S3... {cache_key}, {temp_path}")
                    self._upload_to_s3(cache_key, temp_path)
                if not cache_path.exists():
                    temp_path.replace(cache_path)
            finally:
                # Clean up temp file if it still exists (e.g. replace succeeded
                # or an error occurred before replace)
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.warning(f"Failed to save cache for {video_path}: {e}")

    def clear(self):
        """Clear all cached videos."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared video cache: {self.cache_dir}")

    def get_cache_stats(self):
        """Get cache statistics."""
        cache_files = list(self.cache_dir.rglob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            'num_cached': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_global_cache = None


def get_video_cache():
    """Get or create global video cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = VideoCache()
    return _global_cache
