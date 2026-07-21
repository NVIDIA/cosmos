import ast
import json
from typing import Any, Dict, List


def safe_parse_answer(answer_str: str, verbose: bool = False):
    """
    Safely parse answer string that could be JSON or Python list format
    """
    if answer_str.strip().lower() == "no":
        return None

    # First try JSON parsing
    try:
        return json.loads(answer_str)
    except json.JSONDecodeError:
        if verbose:
            print(f"JSON parsing failed, trying literal evaluation for: {repr(answer_str)}")

        # Try ast.literal_eval for safe evaluation of Python literals
        try:
            result = ast.literal_eval(answer_str)
            if verbose:
                print(f"Literal evaluation successful: {result}")
            return result
        except (ValueError, SyntaxError) as e:
            if verbose:
                print(f"Literal evaluation failed: {e}")
            return None


class HamiltonianPathEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is a valid Hamiltonian path

        Args:
            predicted_answer: Predicted path, could be a list (e.g., [4,5,3,1,2,0]) or string "No"
            ground_truth: Ground truth answer (for reference)
            initial_state: Graph adjacency list representation, format like {'0': [2], '1': [2, 3], '2': [0, 1], ...}

        Returns:
            bool: Whether the predicted answer is correct
        """
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph = safe_parse_answer(initial_state, self.verbose)
            else:
                graph = initial_state

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                predicted_path = safe_parse_answer(predicted_answer, self.verbose)
                if predicted_path is None and predicted_answer.strip().lower() != "no":
                    if self.verbose:
                        print(f"❌ Hamiltonian Path: Cannot parse predicted answer '{predicted_answer}'")
                    return False
            else:
                predicted_path = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                expected_path = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_path = ground_truth

            # If ground truth is "No", check if predicted answer is also "No"
            if expected_path is None:
                if predicted_path is not None:
                    if self.verbose:
                        print(f"❌ Hamiltonian Path: Ground truth is 'No', but predicted answer is {predicted_path}")
                    return False
                return True

            # If predicted answer is "No" but ground truth is not, then error
            if predicted_path is None:
                if self.verbose:
                    print(f"❌ Hamiltonian Path: Predicted answer is 'No', but ground truth is {expected_path}")
                return False

            # Validate if predicted path is a valid Hamiltonian path
            validation_result = self._is_valid_hamiltonian_path(predicted_path, graph)
            if not validation_result:
                return False

            if self.verbose:
                print(f"✅ Hamiltonian Path Evaluation Passed: {predicted_path}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Hamiltonian Path Evaluation Exception: {str(e)}")
            return False

    def _is_valid_hamiltonian_path(self, path: List, graph: Dict) -> bool:
        """
        Validate if the given path is a valid Hamiltonian path
        """
        if not path or not isinstance(path, list):
            if self.verbose:
                print("❌ Hamiltonian Path: Path is empty or not a list")
            return False

        # Get all nodes in the graph
        all_nodes = set()
        for node in graph.keys():
            all_nodes.add(str(node))
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                all_nodes.add(str(neighbor))

        # Convert path nodes to strings (for consistency)
        path_str = [str(node) for node in path]

        # Check if path visits all nodes exactly once
        if set(path_str) != all_nodes:
            if self.verbose:
                missing_nodes = all_nodes - set(path_str)
                extra_nodes = set(path_str) - all_nodes
                if missing_nodes:
                    print(f"❌ Hamiltonian Path: Path missing nodes {missing_nodes}")
                if extra_nodes:
                    print(f"❌ Hamiltonian Path: Path contains non-existent nodes {extra_nodes}")
            return False

        if len(path_str) != len(set(path_str)):
            if self.verbose:
                duplicates = [node for node in path_str if path_str.count(node) > 1]
                print(f"❌ Hamiltonian Path: Path contains duplicate nodes {set(duplicates)}")
            return False

        # Check if adjacent nodes in path are connected in the graph
        for i in range(len(path) - 1):
            current_node = str(path[i])
            next_node = str(path[i + 1])

            # Check if there's an edge from current_node to next_node
            if current_node not in graph:
                if self.verbose:
                    print(f"❌ Hamiltonian Path: Node {current_node} does not exist in graph")
                return False

            neighbors = [str(neighbor) for neighbor in graph[current_node]]
            if next_node not in neighbors:
                if self.verbose:
                    print(f"❌ Hamiltonian Path: No edge between nodes {current_node} and {next_node}")
                return False

        return True


class HamiltonianCycleEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is a valid Hamiltonian cycle

        Args:
            predicted_answer: Predicted cycle, could be a list (e.g., [0,1,2,3,0]) or string "No"
            ground_truth: Ground truth answer (for reference)
            initial_state: Graph adjacency list representation, format like {'0': [2], '1': [2, 3], '2': [0, 1], ...}

        Returns:
            bool: Whether the predicted answer is correct
        """
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph = safe_parse_answer(initial_state, self.verbose)
            else:
                graph = initial_state

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                predicted_cycle = safe_parse_answer(predicted_answer, self.verbose)
                if predicted_cycle is None and predicted_answer.strip().lower() != "no":
                    if self.verbose:
                        print(f"❌ Hamiltonian Cycle: Cannot parse predicted answer '{predicted_answer}'")
                    return False
            else:
                predicted_cycle = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                expected_cycle = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_cycle = ground_truth

            # If ground truth is "No", check if predicted answer is also "No"
            if expected_cycle is None:
                if predicted_cycle is not None:
                    if self.verbose:
                        print(f"❌ Hamiltonian Cycle: Ground truth is 'No', but predicted answer is {predicted_cycle}")
                    return False
                return True

            # If predicted answer is "No" but ground truth is not, then error
            if predicted_cycle is None:
                if self.verbose:
                    print(f"❌ Hamiltonian Cycle: Predicted answer is 'No', but ground truth is {expected_cycle}")
                return False

            # Validate if predicted cycle is a valid Hamiltonian cycle
            if not self._is_valid_hamiltonian_cycle(predicted_cycle, graph):
                return False

            if self.verbose:
                print(f"✅ Hamiltonian Cycle Evaluation Passed: {predicted_cycle}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Hamiltonian Cycle Evaluation Exception: {str(e)}")
            return False

    def _is_valid_hamiltonian_cycle(self, cycle: List, graph: Dict) -> bool:
        """
        Validate if the given cycle is a valid Hamiltonian cycle
        """
        if not cycle or not isinstance(cycle, list) or len(cycle) < 2:
            if self.verbose:
                print("❌ Hamiltonian Cycle: Cycle is empty, not a list, or too short")
            return False

        # Get all nodes in the graph
        all_nodes = set()
        for node in graph.keys():
            all_nodes.add(str(node))
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                all_nodes.add(str(neighbor))

        # Convert cycle nodes to strings (for consistency)
        cycle_str = [str(node) for node in cycle]

        # Check if first and last nodes are the same (forms a cycle)
        if cycle_str[0] == cycle_str[-1]:
            cycle_without_last = cycle_str[:-1]
        else:
            cycle_without_last = cycle_str

        # Check if all nodes are visited exactly once (excluding the duplicate last node)
        if set(cycle_without_last) != all_nodes:
            if self.verbose:
                missing_nodes = all_nodes - set(cycle_without_last)
                extra_nodes = set(cycle_without_last) - all_nodes
                if missing_nodes:
                    print(f"❌ Hamiltonian Cycle: Cycle missing nodes {missing_nodes}")
                if extra_nodes:
                    print(f"❌ Hamiltonian Cycle: Cycle contains non-existent nodes {extra_nodes}")
            return False

        if len(cycle_without_last) != len(set(cycle_without_last)):
            if self.verbose:
                duplicates = [node for node in cycle_without_last if cycle_without_last.count(node) > 1]
                print(f"❌ Hamiltonian Cycle: Cycle contains duplicate nodes {set(duplicates)}")
            return False

        # Check if adjacent nodes in cycle are connected in the graph
        for i in range(len(cycle_str) - 1):
            current_node = cycle_str[i]
            next_node = cycle_str[i + 1]

            # Check if there's an edge from current_node to next_node
            if current_node not in graph:
                if self.verbose:
                    print(f"❌ Hamiltonian Cycle: Node {current_node} does not exist in graph")
                return False

            neighbors = [str(neighbor) for neighbor in graph[current_node]]
            if next_node not in neighbors:
                if self.verbose:
                    print(f"❌ Hamiltonian Cycle: No edge between nodes {current_node} and {next_node}")
                return False

        # For cycles that don't repeat the first node at the end (e.g., [0,1,2,3] instead of [0,1,2,3,0])
        # we need to check if the last node connects back to the first node
        if cycle_str[0] != cycle_str[-1]:
            last_node = cycle_str[-1]
            first_node = cycle_str[0]

            if last_node not in graph:
                if self.verbose:
                    print(f"❌ Hamiltonian Cycle: Last node {last_node} does not exist in graph")
                return False

            neighbors = [str(neighbor) for neighbor in graph[last_node]]
            if first_node not in neighbors:
                if self.verbose:
                    print(f"❌ Hamiltonian Cycle: No edge between last node {last_node} and first node {first_node}")
                return False

        return True


class EulerianPathEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is a valid Eulerian path

        Args:
            predicted_answer: Predicted path, could be a list or string "No"
            ground_truth: Ground truth answer (for reference)
            initial_state: Graph adjacency list representation

        Returns:
            bool: Whether the predicted answer is correct
        """
        if self.verbose:
            print("predicted_answer: ", predicted_answer[:255])
            print("ground_truth: ", ground_truth[:255])
            print("initial_state: ", initial_state)
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph = safe_parse_answer(initial_state, self.verbose)
            else:
                graph = initial_state

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                predicted_path = safe_parse_answer(predicted_answer, self.verbose)
                if predicted_path is None and predicted_answer.strip().lower() != "no":
                    if self.verbose:
                        print(f"❌ Eulerian Path: Cannot parse predicted answer '{predicted_answer}'")
                    return False
            else:
                predicted_path = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                expected_path = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_path = ground_truth

            # If ground truth is "No", check if predicted answer is also "No"
            if expected_path is None:
                if predicted_path is not None:
                    if self.verbose:
                        print(f"❌ Eulerian Path: Ground truth is 'No', but predicted answer is {predicted_path[:120]}")
                    return False
                return True

            # If predicted answer is "No" but ground truth is not, then error
            if predicted_path is None:
                if self.verbose:
                    print(f"❌ Eulerian Path: Predicted answer is 'No', but ground truth is {expected_path[:120]}")
                return False

            # Validate if predicted path is a valid Eulerian path
            if not self._is_valid_eulerian_path(predicted_path, graph):
                return False

            if self.verbose:
                print(f"✅ Eulerian Path Evaluation Passed: {predicted_path}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Eulerian Path Evaluation Exception: {str(e)}")
            return False

    def _is_valid_eulerian_path(self, path: List, graph: Dict) -> bool:
        """
        Validate if the given path is a valid Eulerian path (traverses each edge exactly once)
        """
        if not path or not isinstance(path, list) or len(path) < 2:
            if self.verbose:
                print("❌ Eulerian Path: Path is empty, not a list, or too short")
            return False

        # Build set of edges
        edges = set()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                # For undirected graphs, standardize edges (smaller node first)
                edge = tuple(sorted([str(node), str(neighbor)]))
                edges.add(edge)

        # Check edges in the path
        path_edges = set()
        path_str = [str(node) for node in path]

        for i in range(len(path_str) - 1):
            current_node = path_str[i]
            next_node = path_str[i + 1]
            edge = tuple(sorted([current_node, next_node]))

            # Check if edge exists
            if edge not in edges:
                if self.verbose:
                    print(f"❌ Eulerian Path: Edge ({current_node}, {next_node}) does not exist in graph")
                return False

            # Check if edge has already been used
            if edge in path_edges:
                if self.verbose:
                    print(f"❌ Eulerian Path: Edge ({current_node}, {next_node}) is used more than once")
                return False

            path_edges.add(edge)

        # Check if all edges are traversed
        if path_edges != edges:
            if self.verbose:
                missing_edges = edges - path_edges
                extra_edges = path_edges - edges
                if missing_edges:
                    print(f"❌ Eulerian Path: Path missing edges {missing_edges}")
                if extra_edges:
                    print(f"❌ Eulerian Path: Path contains non-existent edges {extra_edges}")
            return False

        return True


class EulerianCycleEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is a valid Eulerian cycle

        Args:
            predicted_answer: Predicted cycle, could be a list or string "No"
            ground_truth: Ground truth answer (for reference)
            initial_state: Graph adjacency list representation

        Returns:
            bool: Whether the predicted answer is correct
        """
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph = safe_parse_answer(initial_state, self.verbose)
            else:
                graph = initial_state

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                predicted_cycle = safe_parse_answer(predicted_answer, self.verbose)
                if predicted_cycle is None and predicted_answer.strip().lower() != "no":
                    if self.verbose:
                        print(f"❌ Eulerian Cycle: Cannot parse predicted answer '{predicted_answer}'")
                    return False
            else:
                predicted_cycle = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                expected_cycle = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_cycle = ground_truth

            # If ground truth is "No", check if predicted answer is also "No"
            if expected_cycle is None:
                if predicted_cycle is not None:
                    if self.verbose:
                        print(
                            f"❌ Eulerian Cycle: Ground truth is 'No', but predicted answer is {predicted_cycle[:120]}"
                        )
                    return False
                return True

            # If predicted answer is "No" but ground truth is not, then error
            if predicted_cycle is None:
                if self.verbose:
                    print(
                        f"❌ Eulerian Cycle: Predicted answer is 'No', but ground truth is {expected_cycle[:120]}"
                    )
                return False

            # Validate if predicted cycle is a valid Eulerian cycle
            if not self._is_valid_eulerian_cycle(predicted_cycle, graph):
                return False

            if self.verbose:
                print(f"✅ Eulerian Cycle Evaluation Passed: {predicted_cycle}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Eulerian Cycle Evaluation Exception: {str(e)}")
            return False

    def _is_valid_eulerian_cycle(self, cycle: List, graph: Dict) -> bool:
        """
        Validate if the given cycle is a valid Eulerian cycle (traverses each edge exactly once and returns to start)
        """
        if not cycle or not isinstance(cycle, list) or len(cycle) < 3:
            if self.verbose:
                print("❌ Eulerian Cycle: Cycle is empty, not a list, or too short")
            return False

        cycle_str = [str(node) for node in cycle]

        # Check if it forms a cycle (start and end nodes are the same)
        # If not, we need to check if the last node connects back to the first node
        if cycle_str[0] != cycle_str[-1]:
            # For cycles that don't repeat the first node at the end (e.g., [0,1,2] instead of [0,1,2,0])
            # we need to verify that the last node connects back to the first node
            last_node = cycle_str[-1]
            first_node = cycle_str[0]

            if last_node not in graph:
                if self.verbose:
                    print(f"❌ Eulerian Cycle: Last node {last_node} does not exist in graph")
                return False

            neighbors = [str(neighbor) for neighbor in graph[last_node]]
            if first_node not in neighbors:
                if self.verbose:
                    print(f"❌ Eulerian Cycle: No edge between last node {last_node} and first node {first_node}")
                return False

            # Create a new cycle with the first node repeated at the end for validation
            extended_cycle = cycle + [cycle[0]]
            return self._is_valid_eulerian_path_internal(extended_cycle, graph)

        # Use Eulerian path validation logic
        return self._is_valid_eulerian_path_internal(cycle, graph)

    def _is_valid_eulerian_path_internal(self, path: List, graph: Dict) -> bool:
        """
        Internal method: Validate Eulerian path (for use by Eulerian cycle)
        """
        if not path or not isinstance(path, list) or len(path) < 2:
            if self.verbose:
                print("❌ Eulerian Path Internal: Path is empty, not a list, or too short")
            return False

        # Build set of edges
        edges = set()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                edge = tuple(sorted([str(node), str(neighbor)]))
                edges.add(edge)

        # Check edges in the path
        path_edges = set()
        path_str = [str(node) for node in path]

        for i in range(len(path_str) - 1):
            current_node = path_str[i]
            next_node = path_str[i + 1]
            edge = tuple(sorted([current_node, next_node]))

            if edge not in edges:
                if self.verbose:
                    print(f"❌ Eulerian Path Internal: Edge ({current_node}, {next_node}) does not exist in graph")
                return False

            if edge in path_edges:
                if self.verbose:
                    print(f"❌ Eulerian Path Internal: Edge ({current_node}, {next_node}) is used more than once")
                return False

            path_edges.add(edge)

        if path_edges != edges:
            if self.verbose:
                missing_edges = edges - path_edges
                extra_edges = path_edges - edges
                if missing_edges:
                    print(f"❌ Eulerian Path Internal: Path missing edges {missing_edges}")
                if extra_edges:
                    print(f"❌ Eulerian Path Internal: Path contains non-existent edges {extra_edges}")
            return False

        return True


class ConnectivityEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is correct for connectivity problem

        Args:
            predicted_answer: Predicted answer, could be a path list (e.g., [0,3,7,9]) or number (e.g., "3")
            ground_truth: Ground truth answer (for reference)
            initial_state: Graph information including adjacency list, start_node, target_node

        Returns:
            bool: Whether the predicted answer is correct
        """
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph_info = safe_parse_answer(initial_state, self.verbose)
            else:
                graph_info = initial_state

            # Extract graph components
            if isinstance(graph_info, dict) and 'adjacency_list' in graph_info:
                # New format with complete graph info
                adjacency_list = graph_info['adjacency_list']
                start_node = graph_info.get('start_node')
                target_node = graph_info.get('target_node')
            else:
                # Old format - just adjacency list
                adjacency_list = graph_info
                start_node = None
                target_node = None

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                try:
                    # Try to parse as integer (connected components count)
                    predicted_result = int(predicted_answer)
                except ValueError:
                    # Try to parse as list (path)
                    predicted_result = safe_parse_answer(predicted_answer, self.verbose)
                    if predicted_result is None:
                        if self.verbose:
                            print(f"❌ Connectivity: Cannot parse predicted answer '{predicted_answer}'")
                        return False
            else:
                predicted_result = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                try:
                    # Try to parse as integer (connected components count)
                    expected_result = int(ground_truth)
                except ValueError:
                    # Try to parse as list (path)
                    expected_result = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_result = ground_truth

            # Case 1: Both are integers (connected components count)
            if isinstance(predicted_result, int) and isinstance(expected_result, int):
                if predicted_result == expected_result:
                    # Also verify that the predicted count is actually correct
                    actual_components = self._count_connected_components(adjacency_list)
                    if predicted_result == actual_components:
                        if self.verbose:
                            print(
                                "✅ Connectivity Evaluation Passed"
                            )
                        return True
                    else:
                        if self.verbose:
                            print(
                                f"❌ Connectivity: Predicted {predicted_result} components, \
                                but actual count is {actual_components}"
                            )
                        return False
                else:
                    if self.verbose:
                        print(
                            f"❌ Connectivity: Predicted {predicted_result} components, expected {expected_result}"
                        )
                    return False

            # Case 2: Both are lists (paths)
            elif isinstance(predicted_result, list) and isinstance(expected_result, list):
                # Validate the predicted path
                if start_node is not None and target_node is not None:
                    if self._is_valid_path(predicted_result, adjacency_list, start_node, target_node):
                        if self.verbose:
                            print(f"✅ Connectivity Evaluation Passed: Valid path {predicted_result}")
                        return True
                    else:
                        return False
                else:
                    # If start/target nodes are not provided, just check if it's a valid path in the graph
                    if self._is_valid_path_general(predicted_result, adjacency_list):
                        if self.verbose:
                            print(f"✅ Connectivity Evaluation Passed: Valid path {predicted_result}")
                        return True
                    else:
                        return False

            # Case 3: Type mismatch
            else:
                if self.verbose:
                    print(
                        f"❌ Connectivity: Type mismatch. "
                        f"Predicted: {type(predicted_result)}, Expected: {type(expected_result)}"
                    )
                return False

        except Exception as e:
            if self.verbose:
                print(f"❌ Connectivity Evaluation Exception: {str(e)}")
            return False

    def _count_connected_components(self, adjacency_list: Dict) -> int:
        """Count the number of connected components in the graph"""
        visited = set()
        components = 0

        # Get all nodes
        all_nodes = set()
        for node in adjacency_list.keys():
            all_nodes.add(str(node))
        for node, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                all_nodes.add(str(neighbor))

        for node in all_nodes:
            if node not in visited:
                components += 1
                # DFS to mark all nodes in this component
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        # Add unvisited neighbors
                        current_neighbors = adjacency_list.get(current, [])
                        for neighbor in current_neighbors:
                            neighbor_str = str(neighbor)
                            if neighbor_str not in visited:
                                stack.append(neighbor_str)

        return components

    def _is_valid_path(self, path: List, adjacency_list: Dict, start_node: Any, target_node: Any) -> bool:
        """Validate if the path is valid and connects start_node to target_node"""
        if not path or not isinstance(path, list) or len(path) < 1:
            if self.verbose:
                print("❌ Connectivity Path: Path is empty or invalid")
            return False

        path_str = [str(node) for node in path]
        start_str = str(start_node)
        target_str = str(target_node)

        # Check if path starts with start_node and ends with target_node
        if path_str[0] != start_str:
            if self.verbose:
                print(f"❌ Connectivity Path: Path starts with {path_str[0]}, expected {start_str}")
            return False

        if path_str[-1] != target_str:
            if self.verbose:
                print(f"❌ Connectivity Path: Path ends with {path_str[-1]}, expected {target_str}")
            return False

        # Check if adjacent nodes in path are connected
        for i in range(len(path) - 1):
            current_node = str(path[i])
            next_node = str(path[i + 1])

            if current_node not in adjacency_list:
                if self.verbose:
                    print(f"❌ Connectivity Path: Node {current_node} not in graph")
                return False

            neighbors = [str(neighbor) for neighbor in adjacency_list[current_node]]
            if next_node not in neighbors:
                if self.verbose:
                    print(f"❌ Connectivity Path: No edge between {current_node} and {next_node}")
                return False

        return True

    def _is_valid_path_general(self, path: List, adjacency_list: Dict) -> bool:
        """Validate if the path is valid in the graph (without specific start/end requirements)"""
        if not path or not isinstance(path, list) or len(path) < 1:
            if self.verbose:
                print("❌ Connectivity Path: Path is empty or invalid")
            return False

        # Check if adjacent nodes in path are connected
        for i in range(len(path) - 1):
            current_node = str(path[i])
            next_node = str(path[i + 1])

            if current_node not in adjacency_list:
                if self.verbose:
                    print(f"❌ Connectivity Path: Node {current_node} not in graph")
                return False

            neighbors = [str(neighbor) for neighbor in adjacency_list[current_node]]
            if next_node not in neighbors:
                if self.verbose:
                    print(f"❌ Connectivity Path: No edge between {current_node} and {next_node}")
                return False

        return True


class TopologicalSortEvaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract_array(self, text):
        """
        从文本中提取数组，支持嵌套数组和单个数组的情况

        参数:
            text: 包含数组的字符串，如"[1,2,3]"或"[[1,2,3]]"

        返回:
            提取出的数组对象
        """
        import ast
        import re

        # 确保输入是字符串类型
        if not isinstance(text, str):
            return text

        # 去除可能的空白字符
        text = text.strip()

        try:
            # 尝试直接解析
            parsed = ast.literal_eval(text)

            # 如果是嵌套数组且只有一个元素，返回内部数组
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                return parsed[0]

            # 如果是普通数组或其他情况，直接返回
            return parsed
        except (SyntaxError, ValueError):
            # 使用正则表达式尝试提取
            array_pattern = r'\[(.*)\]'
            match = re.search(array_pattern, text)

            if match:
                inner_content = match.group(1).strip()

                # 检查是否是嵌套数组
                if inner_content.startswith('[') and inner_content.endswith(']'):
                    try:
                        return ast.literal_eval(inner_content)
                    except Exception:
                        pass

                # 尝试作为单个数组解析
                try:
                    return ast.literal_eval(f'[{inner_content}]')
                except Exception:
                    # 如果还是失败，可能是格式不规范，使用更宽松的解析方式
                    numbers = re.findall(r'-?\d+', inner_content)
                    if numbers:
                        return [int(num) for num in numbers]

        return None

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate if the predicted answer is a valid topological sort

        Args:
            predicted_answer: Predicted sort order, could be a list or string "No"
            ground_truth: Ground truth answer (for reference)
            initial_state: Directed graph adjacency list representation

        Returns:
            bool: Whether the predicted answer is correct
        """
        # predicted_answer = self.extract_array(predicted_answer)
        if self.verbose:
            print("predicted_answer: ", predicted_answer[:255])
            print("ground_truth: ", ground_truth[:255])
            print("initial_state: ", initial_state)

        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                graph = safe_parse_answer(initial_state, self.verbose)
            else:
                graph = initial_state

            # Parse predicted answer
            if isinstance(predicted_answer, str):
                predicted_sort = safe_parse_answer(predicted_answer, self.verbose)
                if predicted_sort is None and predicted_answer.strip().lower() != "no":
                    if self.verbose:
                        print(f"❌ Topological Sort: Cannot parse predicted answer '{predicted_answer}'")
                    return False
            else:
                predicted_sort = predicted_answer

            # Parse ground truth
            if isinstance(ground_truth, str):
                expected_sort = safe_parse_answer(ground_truth, self.verbose)
            else:
                expected_sort = ground_truth

            # If ground truth is "No", check if predicted answer is also "No"
            if expected_sort is None:
                if predicted_sort is not None:
                    if self.verbose:
                        print(
                            f"❌ Topological Sort: Ground truth is 'No', "
                            f"but predicted answer is {predicted_sort[:120]}"
                        )
                    return False
                return True

            # If predicted answer is "No" but ground truth is not, then error
            if predicted_sort is None:
                if self.verbose:
                    print(f"❌ Topological Sort: Predicted answer is 'No', but ground truth is {expected_sort[:120]}")
                return False

            # Validate if predicted sort is a valid topological sort
            if not self._is_valid_topological_sort(predicted_sort, graph):
                return False

            if self.verbose:
                print(f"✅ Topological Sort Evaluation Passed: {predicted_sort}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Topological Sort Evaluation Exception: {str(e)}")
            return False

    def _is_valid_topological_sort(self, sort_order: List, graph: Dict) -> bool:
        """
        Validate if the given order is a valid topological sort
        """
        if not sort_order or not isinstance(sort_order, list):
            if self.verbose:
                print("❌ Topological Sort: Sort order is empty or not a list")
            return False

        # Get all nodes in the graph
        all_nodes = set()
        for node in graph.keys():
            all_nodes.add(str(node))
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                all_nodes.add(str(neighbor))

        sort_str = [str(node) for node in sort_order]

        # Check if all nodes are included and no duplicates
        if set(sort_str) != all_nodes:
            if self.verbose:
                missing_nodes = all_nodes - set(sort_str)
                extra_nodes = set(sort_str) - all_nodes
                if missing_nodes:
                    print(f"❌ Topological Sort: Sort missing nodes {missing_nodes}")
                if extra_nodes:
                    print(f"❌ Topological Sort: Sort contains non-existent nodes {extra_nodes}")
            return False

        if len(sort_str) != len(set(sort_str)):
            if self.verbose:
                duplicates = [node for node in sort_str if sort_str.count(node) > 1]
                print(f"❌ Topological Sort: Sort contains duplicate nodes {set(duplicates)}")
            return False

        # Create node position mapping
        position = {node: i for i, node in enumerate(sort_str)}

        # Check if all directed edges satisfy topological order
        for node, neighbors in graph.items():
            node_str = str(node)
            for neighbor in neighbors:
                neighbor_str = str(neighbor)
                # For directed edge node -> neighbor, node should come before neighbor
                if position[node_str] >= position[neighbor_str]:
                    if self.verbose:
                        print(
                            f"❌ Topological Sort: Edge ({node_str} -> {neighbor_str}) "
                            f"violates topological order. Position of {node_str}: {position[node_str]}, "
                            f"Position of {neighbor_str}: {position[neighbor_str]}"
                        )
                    return False

        return True
