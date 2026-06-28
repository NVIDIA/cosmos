from typing import List, Tuple

import numpy as np


def calculate_bootstrap_ci(
    test_scores: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    splits: List[int] = None,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for test scores, respecting category splits.

    Args:
        test_scores: List of test scores (0.0 to 1.0 for each test)
        n_bootstrap: Number of bootstrap samples to generate
        ci_level: Confidence interval level (default: 0.95 for 95% CI)
        splits: List of sizes for each category. If provided, resampling will be done
                within each category independently, and the overall score will be the
                average of per-category scores. If None, resampling is done across all tests.

    Returns:
        Tuple of (lower_bound, upper_bound) representing the confidence interval
    """
    if not test_scores:
        return (0.0, 0.0)

    # Convert to numpy array for efficiency
    scores = np.array(test_scores)

    # Simple case - no splits provided, use traditional bootstrap
    if splits is None:
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
    else:
        # Validate splits
        if sum(splits) != len(scores):
            raise ValueError(f"Sum of splits ({sum(splits)}) must equal length of test_scores ({len(scores)})")

        # Convert flat scores list to a list of category scores
        category_scores = []
        start_idx = 0
        for split_size in splits:
            category_scores.append(scores[start_idx: start_idx + split_size])
            start_idx += split_size

        # Generate bootstrap samples respecting category structure
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample within each category independently
            category_means = []
            for cat_scores in category_scores:
                if len(cat_scores) > 0:
                    # Sample with replacement within this category
                    cat_sample = np.random.choice(cat_scores, size=len(cat_scores), replace=True)
                    category_means.append(np.mean(cat_sample))

            # Overall score is average of category means (if any categories have scores)
            if category_means:
                bootstrap_means.append(np.mean(category_means))

    # Calculate confidence interval
    alpha = (1 - ci_level) / 2
    lower_bound = np.percentile(bootstrap_means, alpha * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower_bound, upper_bound)


def perform_permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    splits_a: List[int] = None,
    splits_b: List[int] = None,
) -> Tuple[float, float]:
    """
    Perform a permutation test to determine if there's a significant difference
    between two sets of test scores.

    Args:
        scores_a: List of test scores for candidate A
        scores_b: List of test scores for candidate B
        n_permutations: Number of permutations to perform
        splits_a: List of sizes for each category in scores_a
        splits_b: List of sizes for each category in scores_b

    Returns:
        Tuple of (observed_difference, p_value)
    """
    if not scores_a or not scores_b:
        return (0.0, 1.0)

    # Function to calculate mean of means with optional category splits
    def mean_of_category_means(scores, splits=None):
        if splits is None:
            return np.mean(scores)

        category_means = []
        start_idx = 0
        for split_size in splits:
            if split_size > 0:
                category_scores = scores[start_idx: start_idx + split_size]
                category_means.append(np.mean(category_scores))
            start_idx += split_size

        return np.mean(category_means) if category_means else 0.0

    # Calculate observed difference in means using category structure if provided
    mean_a = mean_of_category_means(scores_a, splits_a)
    mean_b = mean_of_category_means(scores_b, splits_b)
    observed_diff = mean_a - mean_b

    # If no splits are provided, fall back to traditional permutation test
    if splits_a is None and splits_b is None:
        # Combine all scores
        combined = np.concatenate([scores_a, scores_b])
        n_a = len(scores_a)

        # Perform permutation test
        count_greater_or_equal = 0
        for _ in range(n_permutations):
            # Shuffle the combined array
            np.random.shuffle(combined)

            # Split into two groups of original sizes
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]

            # Calculate difference in means
            perm_diff = np.mean(perm_a) - np.mean(perm_b)

            # Count how many permuted differences are >= to observed difference in absolute value
            if abs(perm_diff) >= abs(observed_diff):
                count_greater_or_equal += 1
    else:
        # For category-based permutation test, we need to maintain category structure
        # Validate that the splits match the score lengths
        if splits_a is not None and sum(splits_a) != len(scores_a):
            raise ValueError(f"Sum of splits_a ({sum(splits_a)}) must equal length of scores_a ({len(scores_a)})")
        if splits_b is not None and sum(splits_b) != len(scores_b):
            raise ValueError(f"Sum of splits_b ({sum(splits_b)}) must equal length of scores_b ({len(scores_b)})")

        # Create category structures
        categories_a = []
        categories_b = []

        if splits_a is not None:
            start_idx = 0
            for split_size in splits_a:
                categories_a.append(scores_a[start_idx: start_idx + split_size])
                start_idx += split_size
        else:
            # If no splits for A, treat all scores as one category
            categories_a = [scores_a]

        if splits_b is not None:
            start_idx = 0
            for split_size in splits_b:
                categories_b.append(scores_b[start_idx: start_idx + split_size])
                start_idx += split_size
        else:
            # If no splits for B, treat all scores as one category
            categories_b = [scores_b]

        # Perform permutation test maintaining category structure
        count_greater_or_equal = 0
        for _ in range(n_permutations):
            # For each category pair, shuffle and redistribute
            perm_categories_a = []
            perm_categories_b = []

            for cat_a, cat_b in zip(categories_a, categories_b):
                # Combine and shuffle
                combined = np.concatenate([cat_a, cat_b])
                np.random.shuffle(combined)

                # Redistribute maintaining original sizes
                perm_categories_a.append(combined[: len(cat_a)])
                perm_categories_b.append(combined[len(cat_a):])

            # Flatten permuted categories
            perm_a = np.concatenate(perm_categories_a)
            perm_b = np.concatenate(perm_categories_b)

            # Calculate difference in means respecting category structure
            perm_mean_a = mean_of_category_means(perm_a, splits_a)
            perm_mean_b = mean_of_category_means(perm_b, splits_b)
            perm_diff = perm_mean_a - perm_mean_b

            # Count how many permuted differences are >= to observed difference in absolute value
            if abs(perm_diff) >= abs(observed_diff):
                count_greater_or_equal += 1

    # Calculate p-value
    p_value = count_greater_or_equal / n_permutations

    return (observed_diff, p_value)
