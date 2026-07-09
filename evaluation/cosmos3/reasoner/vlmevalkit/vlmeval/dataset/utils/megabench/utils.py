import importlib
from mimetypes import guess_type


def lazy_import(module_name, class_name):
    """Import the module lazily."""

    def importer():
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    return importer


def is_video_file(file_path):
    mime_type, _ = guess_type(file_path)
    if not mime_type:
        return False
    return mime_type.startswith("video")



def prepare_megabench_data(dataset_name, dataset_subset_name):
    """
        Prepare the MEGA-Bench dataset for evaluation.
        Return:
            subset_dataset: The organized data of the specified subset
            all_dataset: The organized data of all tasks, used for evaluation
    """
    from datasets import load_dataset
    if "single_image" in dataset_subset_name:
        core_data = load_dataset(dataset_name, "core_single_image")
        open_data = load_dataset(dataset_name, "open_single_image")
    else:
        core_data = load_dataset(dataset_name, "core")
        open_data = load_dataset(dataset_name, "open")
    core_test_samples = list(core_data["test"])
    organized_core_dataset = organize_hf_dataset(core_test_samples)
    open_test_samples = list(open_data["test"])
    organized_open_dataset = organize_hf_dataset(open_test_samples)
    subset_dataset = organized_core_dataset if "core" in dataset_subset_name else organized_open_dataset
    all_dataset = organized_core_dataset + organized_open_dataset
    return subset_dataset, all_dataset


def organize_hf_dataset(dataset):
    """
    Organize the dataset with task-based manner

    Return:
        organized_dataset: list, each item is a dict, with the following keys:
            - task_name: str
            - task_query_samples: list of dicts, each dict contains the sample information
    """
    task_dict = {}
    for sample in dataset:
        task_name = sample["task_name"]
        if task_name not in task_dict:
            task_dict[task_name] = []
        task_dict[task_name].append(sample)

    organized_dataset = []
    for task_name, samples in task_dict.items():
        organized_dataset.append({
            "task_name": task_name,
            "task_samples": samples
        })

    return organized_dataset
