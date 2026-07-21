import argparse
import copy
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


def hint_type(value):
    if value == "random":
        return "random"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' must be 'random' or an integer")


class PuzzleFactory():
    def __init__(self) -> None:
        self.constraints = []
        self.game_name = "unknown"
        self.size = 0
        # Define dataset split ratios (must sum to 10)
        self.train_ratio = 8
        self.val_ratio = 1
        self.ablation_ratio = 1

    def sample_hints(self, board: List[List[int]], num_sample_hints: int) -> List[List[int]]:
        # Create a new board filled with zeros
        new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        # Sample num_sample_hints cells to keep from the original board
        sampled_cells = random.sample(range(len(board) * len(board[0])), num_sample_hints)
        for cell in sampled_cells:
            row = cell // len(board[0])
            col = cell % len(board[0])
            new_board[row][col] = board[row][col]  # Copy only the sampled cells from original board
        return new_board
    def save_puzzles(self, puzzles: List[Dict[str, Any]], save_path: str = "datasets/", filename: str = None) -> None:
        """
        Save the generated puzzles to JSON files, split into train, val, and ablation sets.
        Splits are based on unique solutions with ratios defined in __init__.
        Val set has different solutions from train, while ablation shares solutions with train.
        """
        if filename is None:
            base_path = f"{save_path}/{self.game_name}_{self.size}x{self.size}_puzzles"
        else:
            base_path = f"{save_path}/{filename.rsplit('.', 1)[0]}"

        # Group puzzles by their solutions
        solution_groups = {}
        for puzzle in puzzles:
            solution_key = str(puzzle['solution'])  # Convert to string for dict key
            if solution_key not in solution_groups:
                solution_groups[solution_key] = []
            solution_groups[solution_key].append(puzzle)

        # Sort groups (common groups first to validation set) by size for better distribution
        sorted_groups = sorted(solution_groups.items(), key=lambda x: len(x[1]), reverse=True)
        # Calculate target sizes based on ratios
        total_puzzles = len(puzzles)
        target_val_size = total_puzzles * self.val_ratio // 10
        target_ablation_size = total_puzzles * self.ablation_ratio // 10
        # Initialize sets
        train_puzzles = []
        val_puzzles = []
        ablation_puzzles = []
        # First, fill validation set with complete groups
        val_solutions = set()
        current_val_size = 0
        val_group_idx = 0
        while val_group_idx < len(sorted_groups) and current_val_size < target_val_size:
            group = sorted_groups[val_group_idx][1]
            if current_val_size + len(group) <= target_val_size * 1.2:  # Allow 20% overflow
                val_puzzles.extend(group)
                val_solutions.add(sorted_groups[val_group_idx][0])
                current_val_size += len(group)
            val_group_idx += 1

        # Fill train and ablation sets with remaining groups
        train_solutions = set()
        current_ablation_size = 0

        for solution, group in sorted_groups:
            if solution in val_solutions:
                continue

            train_solutions.add(solution)
            # Randomly split each remaining group between train and ablation
            if current_ablation_size < target_ablation_size:
                # Calculate how many puzzles we can still add to ablation
                space_left = target_ablation_size - current_ablation_size
                # Take up to 20% of the current group for ablation
                ablation_count = min(max(1, len(group) // 5), space_left)
                # Randomly select puzzles for ablation
                ablation_indices = random.sample(range(len(group)), ablation_count)
                for i in range(len(group)):
                    if i in ablation_indices:
                        ablation_puzzles.append(group[i])
                        current_ablation_size += 1
                    else:
                        train_puzzles.append(group[i])
            else:
                train_puzzles.extend(group)

        # Shuffle each set before saving
        random.shuffle(train_puzzles)
        random.shuffle(val_puzzles)
        random.shuffle(ablation_puzzles)

        # Create all parent directories
        os.makedirs(os.path.dirname(f"{base_path}_train.json"), exist_ok=True)

        # Save splits to separate files
        for split_name, split_puzzles in [
            ("train", train_puzzles),
            ("val", val_puzzles),
            ("ablation", ablation_puzzles)
        ]:
            split_path = f"{base_path}_{split_name}.json"
            with open(split_path, "w") as f:
                json.dump(split_puzzles, f, indent=2)
        print(f"\nSplit and saved {len(puzzles)} puzzles:")
        print(f"Train: {len(train_puzzles)} puzzles ({len(train_solutions)} unique solutions)")
        print(f"Val: {len(val_puzzles)} puzzles ({len(val_solutions)} unique solutions)")
        print(f"Ablation: {len(ablation_puzzles)} puzzles (solutions shared with train)")
        print(f"Files saved to {base_path}_[train/val/ablation].json")

    def check(self, game_state: Dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(game_state):
                return False
        return True

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        pass
