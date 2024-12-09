from synthetic_data.apply_sentence_embedding_reranking import main
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_path', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--output_file', type=str, default='data/cordis/generative/all_synthetic.json')
    arg_parser.add_argument('--db_id', type=str, default='cordis_temporary')

    args = arg_parser.parse_args()
    main(args)