"""
Utility functions for formatting evaluation results in a standardized way.
"""

from typing import Dict, Union

import pandas as pd


def dict_to_rows_df(
    data: Dict[str, Union[float, int]],
    category_col: str = 'Category',
    value_col: str = 'Accuracy'
) -> pd.DataFrame:
    """
    Convert a dictionary of category->value pairs into a DataFrame with rows.

    This standardizes the output format for accuracy/metric CSV files across all datasets.
    Instead of categories as columns, each category becomes a row.

    Args:
        data: Dictionary mapping category names to values, e.g. {'Cat1': 0.5, 'Cat2': 0.6}
        category_col: Name for the category column (default: 'Category')
        value_col: Name for the value column (default: 'Accuracy')

    Returns:
        DataFrame with categories as rows

    Example:
        >>> data = {'Overall': 0.75, 'Type_A': 0.8, 'Type_B': 0.7}
        >>> df = dict_to_rows_df(data)
        >>> print(df)
           Category  Accuracy
        0   Overall      0.75
        1    Type_A      0.80
        2    Type_B      0.70
    """
    rows = [
        {category_col: category, value_col: value}
        for category, value in data.items()
    ]
    return pd.DataFrame(rows)
