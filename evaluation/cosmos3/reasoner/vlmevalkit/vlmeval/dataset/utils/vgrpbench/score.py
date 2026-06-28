"""
VGRPBench scoring module for evaluating visual grid reasoning puzzle solutions.

This module provides functions to evaluate puzzle solutions from language models,
including parsing model outputs, checking perception accuracy, and verifying solutions.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from json_repair import repair_json
from tqdm import tqdm

from . import puzzles
from .puzzles import common_get_game_factory as get_game_factory

# Global variable to store the puzzle grid size
GRID_SIZE = None


def extract_perception_and_answer(model_output):
    """
    Extract both perception and answer from model output.

    Parses the model's output to extract the perceived initial state and the solution.
    Handles different output formats and section headers.

    Args:
        model_output (str): The raw output from the model

    Returns:
        tuple: (initial_state, solution) where both are 2D arrays or None if parsing fails
    """
    try:
        # Handle plain text format
        if "Initial State" in model_output:
            parts = model_output.split('Initial State\n', 1)
        elif "Perception" in model_output:
            parts = model_output.split('Perception\n', 1)
        else:
            return None, None

        if len(parts) != 2:
            return None, None
        content = parts[1]

        if "Answer" in content:
            perception_answer = content.split('\nAnswer\n')
        elif "Solution" in content:
            perception_answer = content.split('\nSolution\n')
        else:
            return None, None

        if len(perception_answer) != 2:
            return None, None

        perception, answer = perception_answer

        if perception.strip() == "Wrong":
            initial_state = None
            # Remove outer brackets and split into rows
            raw_solution = answer.strip()[2:-2].split('],[')
            solution = [[c for c in row.split(',')] for row in raw_solution]
        else:
            if answer.strip() == "Wrong":
                raw_perception = perception.strip()[2:-2].split('],[')
                initial_state = [[c for c in row.split(',')] for row in raw_perception]
                solution = None
            else:
                # Remove outer brackets and split into rows
                raw_perception = perception.strip()[2:-2].split('],[')
                initial_state = [[c for c in row.split(',')] for row in raw_perception]
                raw_solution = answer.strip()[2:-2].split('],[')
                solution = [[c for c in row.split(',')] for row in raw_solution]

        initial_state = [[cell if cell != '*' else 0 for cell in row] for row in initial_state]

        return initial_state, solution
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None, None


def check_perception(thoughts, init_board, game_type):
    """
    Check if model's perception matches the initial board.

    Compares the model's understanding of the initial state with the actual initial state,
    with game-specific adjustments for different puzzle types.

    Args:
        thoughts (list): 2D array representing the model's perception of the initial state
        init_board (list): 2D array representing the actual initial state
        game_type (str): Type of puzzle game
    Returns:
        bool: True if perception matches initial board, False otherwise
    """
    # Game-specific adjustments
    if game_type == "battleships":
        init_board = [[0 if cell == 'e' else cell for cell in row] for row in init_board]
        thoughts = [[0 if cell == 'e' else cell for cell in row] for row in thoughts]
    if game_type == "lightup":
        for i in range(len(init_board)):
            for j in range(len(init_board[i])):
                cell = init_board[i][j]
                # Check if cell is a number (not 0) or not a string/character
                if (isinstance(cell, (int, float)) and cell != 0) or (isinstance(cell, str) and not cell.isalpha()):
                    init_board[i][j] = 'w'
    if game_type == "fieldexplore":
        # Convert -1 to 0 in init_board
        init_board = [[0 if cell == -1 else cell for cell in row] for row in init_board]
    # Convert string representation to 2D grid if needed
    if isinstance(init_board, str):
        init_grid = [[c for c in row] for row in init_board.strip().split('\n')]
    else:
        init_grid = init_board

    # Check dimensions match
    if len(thoughts) != len(init_grid) or any(len(row) != len(init_grid[0]) for row in thoughts):
        return False
    # Check cell by cell
    for i in range(len(init_grid)):
        for j in range(len(init_grid[0])):
            if str(init_grid[i][j]) != str(thoughts[i][j]):
                return False
    return True


def check_answer(answer, init_board, game_factory):
    """
    Verify if the model's answer is correct for the given puzzle.

    Performs game-specific validations and uses the game factory to check solution correctness.

    Args:
        answer (list): 2D array representing the model's solution
        init_board (list): 2D array representing the initial state
        game_factory (GameFactory): Factory object for the specific game type
    Returns:
        bool: True if the answer is correct, False otherwise
    """
    global GRID_SIZE
    # Game-specific preprocessing for answers
    if game_factory.game_name in ["treesandtents", "starbattle", "hitori", "aquarium", "kakurasu"]:
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                if answer[i][j] in [0, '0']:
                    answer[i][j] = 'e'
    if game_factory.game_name == "oddevensudoku":
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                try:
                    answer[i][j] = int(answer[i][j])
                except Exception as e:
                    return False
    if game_factory.game_name == "lightup":
        # Convert '0' to 'e'
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                if answer[i][j] == '0':
                    answer[i][j] = 'e'
    # Convert string representation to 2D grid if needed
    if isinstance(init_board, str):
        init_grid = [[c for c in row] for row in init_board.strip().split('\n')]
    else:
        init_grid = init_board
    # Check dimensions
    if len(answer) != GRID_SIZE or any(len(row) != GRID_SIZE for row in answer):
        return False

    # Game-specific validation for initial values
    if game_factory.game_name == "hitori":
        # Compare with game_factory.additional_board
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if game_factory.additional_board[i][j] not in [0, '0'] and str(game_factory.additional_board[i][j]) != str(answer[i][j]):
                    return False
    elif game_factory.game_name == "nonogram":
        # Convert 0, '0', '*' in answer to 'e'
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if answer[i][j] in [0, '0', '*']:
                    answer[i][j] = 'e'
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if init_grid[i][j] not in [0, '0'] and str(init_grid[i][j]) != str(answer[i][j]):
                    return False
    elif game_factory.game_name == "fieldexplore":
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # 's' on the initial board must be kept
                if init_grid[i][j] == 's' and not answer[i][j] == 's':
                    return False
                try:
                    cell_value = int(init_grid[i][j])
                    if cell_value > 0 and str(answer[i][j]) == 's':
                        return False
                except (ValueError, TypeError):
                    # Cell is not a number, continue with other checks
                    pass
        return True
    else:
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if init_grid[i][j] not in [0, '0', 'e'] and str(init_grid[i][j]) != str(answer[i][j]):
                    return False
    # Prepare game state for validation
    game_state = {
        "board": answer,
        "size": GRID_SIZE,
    }

    # Add game-specific state information
    if game_factory.game_name == "skyscraper":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "coloredsudoku":
        game_state["colors"] = game_factory.current_colors
    elif game_factory.game_name == "futoshiki":
        game_state["inequalities"] = game_factory.current_inequalities
    elif game_factory.game_name == "killersudoku":
        game_state["cages"] = game_factory.cages
    elif game_factory.game_name == "renzoku":
        game_state["hints"] = game_factory.hints
    elif game_factory.game_name == 'kakuro':
        game_state["sums"] = game_factory.current_sums
    elif game_factory.game_name == "thermometers":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "treesandtents":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "starbattle":
        game_state["regions"] = game_factory.regions
    elif game_factory.game_name == "hitori":
        game_state["numbers"] = game_factory.numbers
    elif game_factory.game_name == "aquarium":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "kakurasu":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "oddevensudoku":
        game_state["cell_types"] = game_factory.cell_types
    elif game_factory.game_name == "nonogram":
        game_state["hints"] = game_factory.hints
    elif game_factory.game_name == "lightup":
        game_state["wall_numbers"] = game_factory.wall_numbers
    elif game_factory.game_name == "battleships":
        game_state["hints"] = game_factory.hints
    # Validate the solution using the game factory
    try:
        return game_factory.check(game_state)
    except Exception as e:
        print(f"Error checking answer: {e}")
        return False


def calculate_group_statistics(outcomes, num_groups=5):
    """
    Calculate group-wise means and the standard deviation between groups.

    Splits outcomes into groups and calculates statistics to estimate variance.

    Args:
        outcomes (list): Binary outcomes (0 or 1) for each puzzle
        num_groups (int): Number of groups to split the data into

    Returns:
        tuple: (group_means, group_std) where group_means is a list of percentages
               and group_std is the standard deviation between groups
    """
    if not outcomes:
        return [], 0.0

    # Convert to numpy array for easier manipulation
    outcomes = np.array(outcomes)

    # Calculate number of items per group
    group_size = len(outcomes) // num_groups

    # Split into groups and calculate mean for each group
    group_means = []
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < num_groups - 1 else len(outcomes)
        group = outcomes[start_idx:end_idx]
        group_means.append(np.mean(group) * 100)  # Convert to percentage

    # Calculate standard deviation between group means
    group_std = np.std(group_means)

    return group_means, group_std


def evaluate_single_puzzle(model_output, puzzle_data, game_type):
    """
    Evaluate a single puzzle solution.

    Processes model output and puzzle data to determine if the model correctly
    understood the puzzle and provided a valid solution.

    Args:
        model_output (str): The raw output from the model
        puzzle_data (dict): Puzzle data including initialization
        game_type (str): Type of puzzle game (e.g., "thermometers", "sudoku")
    Returns:
        dict: Evaluation results including perception_correct, answer_correct, and score
    """
    # Add puzzle directory to path if needed
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    puzzle_dir = os.path.join(curr_dir, "puzzles")
    if puzzle_dir not in sys.path:
        sys.path.append(puzzle_dir)

    # Initialize the appropriate game factory for the puzzle type
    GameFactory = get_game_factory.get_game_factory(game_type)

    init_board = puzzle_data['initialization']

    game_factory = GameFactory(size=4)

    # Game-specific initialization handling
    if game_type == "coloredsudoku":
        colors = puzzle_data.get('colors', None)
        game_factory.current_colors = colors
    elif game_type == "binairo":
        init_board = puzzle_data.get('initialization', None)
    elif game_type == "futoshiki":
        row_inequalities = puzzle_data.get('row_inequalities', None)
        col_inequalities = puzzle_data.get('col_inequalities', None)
        game_factory.current_inequalities = {
            "row": row_inequalities,
            "col": col_inequalities
        }
    elif game_type == "killersudoku":
        cages = puzzle_data.get('cages', None)
        game_factory.cages = cages
    elif game_type == "renzoku":
        hints = puzzle_data.get('hints', None)
        game_factory.hints = hints
    elif game_type == "kakuro":
        sums = puzzle_data.get('sums', None)
        game_factory.current_sums = sums
    elif game_type == "skyscraper":
        clues = puzzle_data.get('initialization', None).get('clues')
        init_board = puzzle_data.get('initialization', None).get('board')  # Special case
        game_factory.clues = clues
    elif game_type == "thermometers":
        clues = puzzle_data.get('initialization', None).get('clues')
        game_factory.clues = clues
        init_board = puzzle_data.get('initialization', None).get('board')
    elif game_type == "treesandtents":
        clues = puzzle_data.get('clues', None)
        game_factory.clues = clues
        init_board = puzzle_data.get('initialization', None)
    elif game_type == "starbattle":
        init_board = puzzle_data.get('initialization', None)
        game_factory.regions = puzzle_data.get('regions', None)
    elif game_type == "hitori":
        init_board = puzzle_data.get('initialization').get('numbers', None)
        game_factory.numbers = puzzle_data.get('initialization', None).get('numbers')
        game_factory.additional_board = puzzle_data.get('initialization', None).get('board')
    elif game_type == "aquarium":
        init_board = puzzle_data.get('initialization', None).get('board')
        game_factory.clues = puzzle_data.get('initialization', None).get('clues', None)
    elif game_type == "kakurasu":
        init_board = puzzle_data.get('initialization', None).get('board')
        game_factory.clues = puzzle_data.get('initialization', None).get('clues', None)
    elif game_type == "oddevensudoku":
        game_factory.cell_types = puzzle_data.get('cell_types')
        init_board = puzzle_data.get('initialization', None)
    elif game_type == "battleships":
        init_board = puzzle_data.get('initialization', None)
        game_factory.hints = puzzle_data.get('hints', None)
    elif game_type == "jigsawsudoku":
        init_board = puzzle_data.get('initialization', None)
    elif game_type == "nonogram":
        init_board = puzzle_data.get('initialization', None)
        game_factory.hints = puzzle_data.get('hints', None)
    elif game_type == "lightup":
        init_board = puzzle_data.get('initialization', None)
        game_factory.wall_numbers = puzzle_data.get('wall_numbers', None)
    # Set grid size
    global GRID_SIZE
    GRID_SIZE = len(init_board) if GRID_SIZE is None else GRID_SIZE

    # Extract model's perception and answer from its output
    thoughts, answer = extract_perception_and_answer(model_output)
    # Early return if parsing failed
    if thoughts is None or answer is None:
        return {
            "perception_correct": False,
            "answer_correct": False,
            "number_of_samples": 1
        }

    # Game-specific preprocessing
    try:
        if game_type == "starbattle":
            for i in range(len(thoughts)):
                for j in range(len(thoughts[i])):
                    if thoughts[i][j] == "*":
                        thoughts[i][j] = "0"
    except Exception as e:
        print(f"starbattle: Error converting thoughts to 0: {e}")
    try:
        if game_type == "killersudoku":
            answer = [[int(cell) for cell in row] for row in answer]
    except Exception as e:
        answer = None

    # Special handling for trees and tents
    if game_type == "treesandtents":
        # Convert shorthand symbols to standard format
        for i in range(len(thoughts)):
            for j in range(len(thoughts[i])):
                if thoughts[i][j] == 't':
                    thoughts[i][j] = 'tt'
                elif thoughts[i][j] == 'r':
                    thoughts[i][j] = 'tr'
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                if answer[i][j] == 't':
                    answer[i][j] = 'tt'
                elif answer[i][j] == 'r':
                    answer[i][j] = 'tr'

    # Check perception and answer
    perception_correct = check_perception(thoughts, init_board, game_type)
    answer_correct = check_answer(answer, init_board, game_factory) if perception_correct else False

    return {
        "perception_correct": perception_correct,
        "answer_correct": answer_correct,
        "number_of_samples": 1
    }


if __name__ == "__main__":
    main()
