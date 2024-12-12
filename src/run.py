from synthetic_data.generate_synthetical_data_TrialBench import main
import argparse

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