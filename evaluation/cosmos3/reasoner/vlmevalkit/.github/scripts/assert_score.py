import argparse
import ast
import json
import os

import pandas as pd


def validate_scores(dataset_list, assert_score, model_name):
    for dataset in dataset_list:
        base_score = assert_score[dataset][model_name]
        if dataset == "OCRBench_MINI":
            score_file = os.path.join("outputs", f"{model_name}/{model_name}_{dataset}_score.json")
            cur_score = 0
            with open(score_file, "r") as f:
                total_score = json.load(f)
                cur_score = total_score["Final Score Norm"]
            assert (
                abs(cur_score - float(base_score)) <= 0.01
            ), f"{dataset} on {model_name}: cur_score is {cur_score}, base_score is {base_score}"
        else:
            score_file = os.path.join("outputs", f"{model_name}/{model_name}_{dataset}_acc.csv")
            df = pd.read_csv(score_file)
            cur_score = df["Overall"].iloc[0]
            if dataset == "MMBench_V11_MINI":
                cur_score = df.loc[df["split"] == "dev", "Overall"].values
            assert (
                abs(cur_score - float(base_score)) <= 0.01
            ), f"{dataset} on {model_name}: cur_score is {cur_score}, base_score is {base_score}"
        print(f"cur_score is {cur_score}, base_score is {base_score}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate model scores against csv/json data")

    parser.add_argument("--dataset", type=str, required=True, help="Space-separated list of datasets")

    parser.add_argument(
        "--base_score", type=str, required=True, help="Dictionary string in format {dataset:{model:score}}"
    )

    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to validate")

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        dataset_list = args.dataset.split()
        base_score = ast.literal_eval(args.base_score)
    except Exception as e:
        print(f"Parameter parsing error: {str(e)}")
        return

    validate_scores(dataset_list, base_score, args.model_name)


if __name__ == "__main__":
    main()
