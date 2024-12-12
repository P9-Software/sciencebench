from synthetic_data.generate_synthetical_data_TrialBench import main
import argparse

# all_trial_metadata
# python3 src/run.py --data_path data/TrialBench --output_folder data/TrialBench/generative/generated/all_trial_metadata --database all_trial_metadata --db_path all_trial_metadata.db
# gcmd
# python3 src/run.py --data_path data/TrialBench --output_folder data/TrialBench/generative/generated/gcmd --database gcmd --db_path gcmd.db
# combined
# python3 src/run.py --data_path data/TrialBench --output_folder data/TrialBench/generative/generated/combined --database combined --db_path combined.db
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/TrialBench')
    arg_parser.add_argument('--output_folder', type=str, default='data/TrialBench/generative/generated')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--base_number_of_samples_per_query_type', type=int, default=50, help='The base number of samples per query type. '
                                                                                                 'This number, multiplied with the query type multiplier (see "common_query_types.py") '
                                                                                                 'is the total number of samples that will be generated for each query type.')

    arg_parser.add_argument('--database', type=str, default='all_trial_metadata')
    arg_parser.add_argument('--db_path', type=str,default='all_trial_metadata.db')

    args = arg_parser.parse_args()
    main(args)