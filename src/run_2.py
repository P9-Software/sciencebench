from synthetic_data.apply_sentence_embedding_reranking import main
import argparse

# all_trial_metadata
# python3 src/run_2.py --input_data_path data/TrialBench/generative/generated/all_trial_metadata --output_file data/TrialBench/generative/all_trial_metadata_synthetic.json --db_id all_trial_metadata --model granite-8b
# gcmd
# python3 src/run_2.py --input_data_path data/TrialBench/generative/generated/gcmd --output_file data/TrialBench/generative/gcmd_synthetic.json --db_id gcmd --model granite-8b
# combined
# python3 src/run_2.py --input_data_path data/TrialBench/generative/generated/combined --output_file data/TrialBench/generative/combined_synthetic.json --db_id combined --model granite-8b

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_path', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--output_file', type=str, default='data/cordis/generative/all_synthetic.json')
    arg_parser.add_argument('--db_id', type=str, default='cordis_temporary')

    args = arg_parser.parse_args()
    main(args)