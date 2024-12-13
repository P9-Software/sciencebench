from synthetic_data.generate_synthetical_data_TrialBench import generate_synthetic_data
from synthetic_data.apply_sentence_embedding_reranking import rerank
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--database', type=str)
    args = arg_parser.parse_args()
    models = {
        "granite-instruct-3b": "ibm-granite/granite-3.0-8b-instruct",
        "granite-code-34b": "ibm-granite/granite-34b-code-instruct-8k",
        "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    }

    if args.database == "all_trial_metadata":
        generate_synthetic_data(args.database, "data/TrialBench", "all_trial_metadata.db", args.number_of_choices, models)
        rerank(args.database, "data/TrialBench/generative/generated/all_trial_metadata", "data/TrialBench/generative/all_trial_metadata_synthetic.json")
    elif args.database == "gcmd":
        generate_synthetic_data(args.database, "data/TrialBench", "gcmd.db", args.number_of_choices, models)
        rerank(args.database, "data/TrialBench/generative/generated/gcmd", "data/TrialBench/generative/gcmd_synthetic.json")
    elif args.database == "combined":
        generate_synthetic_data(args.database, "data/TrialBench", "combined.db", args.number_of_choices, models)
        rerank(args.database, "data/TrialBench/generative/generated/combined", "data/TrialBench/generative/combined.json")
    else:
        generate_synthetic_data("all_trial_metadata", "data/TrialBench", "all_trial_metadata.db", args.number_of_choices, models)
        generate_synthetic_data("gcmd", "data/TrialBench", "gcmd.db", args.number_of_choices, models)
        generate_synthetic_data("combined", "data/TrialBench", "combined.db", args.number_of_choices, models)
        rerank(args.database, "data/TrialBench/generative/generated/all_trial_metadata", "data/TrialBench/generative/all_trial_metadata_synthetic.json")
        rerank(args.database, "data/TrialBench/generative/generated/gcmd", "data/TrialBench/generative/gcmd_synthetic.json")
        rerank(args.database, "data/TrialBench/generative/generated/combined", "data/TrialBench/generative/combined.json")
