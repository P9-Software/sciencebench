from synthetic_data.generate_synthetical_data_cordis import main
import argparse

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/cordis')
    arg_parser.add_argument('--output_folder', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--base_number_of_samples_per_query_type', type=int, default=50, help='The base number of samples per query type. '
                                                                                                 'This number, multiplied with the query type multiplier (see "common_query_types.py") '
                                                                                                 'is the total number of samples that will be generated for each query type.')

    arg_parser.add_argument('--database', type=str, default='cordis_temporary')
    arg_parser.add_argument('--db_user', type=str, default='postgres')
    arg_parser.add_argument('--db_password', type=str, default='vdS83DJSQz2xQ')
    arg_parser.add_argument('--db_host', type=str, default='testbed.inode.igd.fraunhofer.de')
    arg_parser.add_argument('--db_port', type=str, default='18001')
    arg_parser.add_argument('--db_options', type=str, default=f"-c search_path=unics_cordis,public")
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='davinci:ft-personal-2022-01-17-10-28-10')

    args = arg_parser.parse_args()
    main(args)