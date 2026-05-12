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

"""
Weighted multi-aspect ratio shard distributor.
Subclass of ShardlistMultiAspectRatio that adds sampling weighted by data source
within each aspect-ratio partition. Preserves per-worker aspect-ratio assignment and DDP behavior.
"""

import os
import random
import time
from datetime import datetime

from webdataset.utils import pytorch_worker_info

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos3._src.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
from cosmos3._src.imaginaire.utils import log


class WeightedShardlistMultiAspectRatio(ShardlistMultiAspectRatio):
    r"""
    Multi-aspect ratio shard list with weighted sampling by data source.
    Each worker still receives URLs for a single aspect ratio (same as base class).
    Within that set, URLs are sampled by datasource according to data_weight_dict.
    """

    def __init__(
        self,
        data_weight_dict: dict | None = None,
        shuffle: bool = True,
        split_by_node: bool = True,
        split_by_worker: bool = True,
        chunk_size: int = 1,
        resume_flag: bool = True,
        verbose: bool = False,
        is_infinite_loader: bool = False,
        dump_worker_category_distribution: bool = False,
    ):
        r"""Create a weighted multi-aspect ratio ShardList.

        Args:
            data_weight_dict (dict | None): Mapping from data source name to weight.
                If None, behaves like ShardlistMultiAspectRatio (no weighting).
            shuffle (bool): Shuffle samples before iterating.
            split_by_node (bool): Split shards by node if True.
            split_by_worker (bool): Split shards by worker if True.
            chunk_size (int): Chunk size used in webdataset creation.
            resume_flag (bool): If enabled, resumes from WDS_EPOCH_NUM and WDS_START_INDEX.
            verbose (bool): Print extra logs if True.
            is_infinite_loader (bool): If True, dataloader runs indefinitely with weighted sampling.
            dump_worker_category_distribution (bool): If True, dump the worker category distribution to one csv file per worker.
        """
        super().__init__(
            shuffle=shuffle,
            split_by_node=split_by_node,
            split_by_worker=split_by_worker,
            chunk_size=chunk_size,
            resume_flag=resume_flag,
            verbose=verbose,
            is_infinite_loader=is_infinite_loader,
        )
        self.data_weight_dict = data_weight_dict
        self.dump_worker_category_distribution = dump_worker_category_distribution
        if self.dump_worker_category_distribution:
            self.weight_per_tar_csv_dir = f"outputs/weight_csvs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.weight_per_tar_csv_dir, exist_ok=True)

    def set_urls(self, urls: list[TarSample]):
        super().set_urls(urls)
        if self.data_weight_dict:
            # Count global *samples* per datasource *before* per-worker splitting so that
            # each tar file can be assigned weight = datasource_weight * global_sample_count.
            global_sample_counts: dict[str, int] = {}
            for url in urls:
                src = url.meta.source
                global_sample_counts[src] = global_sample_counts.get(src, 0) + url.num_samples
            self._global_datasource_sample_counts = global_sample_counts
            for src in global_sample_counts:
                log.info(f"Global counts for {src}: {global_sample_counts[src]} samples")
        if self.verbose:
            # Log aspect-ratio split from base class (ratio feature is used per-worker)
            if hasattr(self, "url_aspect_split") and self.url_aspect_split:
                ratio_summary = {ar: len(entries) for ar, entries in self.url_aspect_split.items()}
                log.info(
                    f"WeightedShardlistMultiAspectRatio: aspect_ratio split (ratio feature active): {ratio_summary}"
                )
            if self.data_weight_dict:
                log.info(f"data_weight_dict: {self.data_weight_dict}")

    def __iter__(self):
        url_list = self.obtain_url_list()

        # Group URLs by datasource within this worker's list
        urls_by_datasource: dict[str, list[TarSample]] = {}
        for url in url_list:
            datasource = url.meta.source
            if datasource not in self.data_weight_dict:
                raise ValueError(
                    f"Datasource '{datasource}' from URL not found in data_weight_dict. "
                    f"Available: {list(self.data_weight_dict.keys())}"
                )
            if datasource not in urls_by_datasource:
                urls_by_datasource[datasource] = []
            urls_by_datasource[datasource].append(url)

        if self.verbose:
            counts = {cat: len(u) for cat, u in urls_by_datasource.items()}
            log.info(
                f"WeightedShardlistMultiAspectRatio: weighted sampling active — "
                f"URLs per datasource (this worker): {counts}, weights={self.data_weight_dict}"
            )

        datasource_names = list(urls_by_datasource.keys())

        if self.is_infinite_loader:
            rank, world_size, worker_id, num_workers = pytorch_worker_info()
            # One RNG per worker, seeded once
            worker_seed = (rank * num_workers + worker_id) + int(time.time() * 10000)
            rng = random.Random(worker_seed)

            # Build a flat list of tar files with per-tar weights.
            # Each tar from datasource C gets weight = data_weight_dict[C] * global_samples_C.
            flat_urls: list[TarSample] = []
            flat_weights: list[float] = []
            if self.dump_worker_category_distribution:
                weight_csv_file = open(
                    os.path.join(self.weight_per_tar_csv_dir, f"_weight_per_tar_{rank * num_workers + worker_id}.csv"),
                    "w",
                )
                weight_csv_file.write(
                    "datasource,wdinfo,path,weight,global_samples,data_list_key_count,data_weight_dict\n"
                )

            for datasource in datasource_names:
                tars = urls_by_datasource[datasource]
                global_samples = self._global_datasource_sample_counts[datasource]
                for url in tars:
                    per_tar_weight = self.data_weight_dict[datasource] / global_samples

                    flat_urls.append(url)
                    flat_weights.append(per_tar_weight)
                    if self.dump_worker_category_distribution:
                        weight_csv_file.write(
                            f"{datasource},{url.meta.wdinfo},{url.path},{per_tar_weight},{global_samples},{url.num_samples},{self.data_weight_dict[datasource]}\n"
                        )
            if self.dump_worker_category_distribution:
                weight_csv_file.close()

            while True:
                url = rng.choices(flat_urls, weights=flat_weights, k=1)[0]
                yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
