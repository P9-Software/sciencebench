import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace


from synthetic_data.common_query_types import all_trial_metadata_query_types, gcmd_query_types
from synthetic_data.sample_queries.sample_query import sample_query
from tools.transform_generative_schema import GenerativeSchema
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK = "text-generation"
MAX_NEW_TOKENS = 500

"""
Synthetic data generator by using Ursin's templates
"""
def main(args):
    device = "cuda"
    model_path = "ibm-granite/granite-3.0-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # drop device_map if running on CPU
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    # change input text as desired
    print(args.database)
    table_file = args.database + '_tables.json'
    with open(Path(args.data_path) / 'original' / table_file) as f:
        schemas = json.load(f)
        original_schema = schemas[0]  # we assume there is only one db-schema in this file
    generated_schema_file = args.database + '_generative_schema.json'
    generative_schema = GenerativeSchema(Path(args.data_path) / 'generative' / generated_schema_file)

    db_config = SimpleNamespace(database=args.database,
                                db_user="",
                                db_password="",
                                db_host="",
                                db_port="",
                                db_options="",
                                path=args.db_path)

    query_cache = []
    query_types = all_trial_metadata_query_types() if args.database == "all_trial_metadata" else gcmd_query_types()

    for idx, (query_type, multiplier) in enumerate(query_types.items()):

        round_idx = 0
        fail_to_sample = 0

        # we might have to repeat the sampling process multiple times to get enough samples (exceptions due to unfavorable samplings),
        # but we still don't want to be caught in an infinite loop.
        while round_idx < (args.base_number_of_samples_per_query_type * multiplier) and fail_to_sample < 50:

            try:
                sampled_query, sampled_query_replaced = sample_query(query_type, original_schema, generative_schema, db_config)

                if sampled_query in query_cache:
                    raise ValueError('Query already sampled')
                else:
                    query_cache.append(sampled_query)

                print(f'{query_type}                        {sampled_query}')

                table_name = find_table_name_in_sample(sampled_query)
                table_string = table_name_lookup(args, table_name)

                prompt = "As an experienced and professional clinical trial data analyst, your task is to create " + str(args.number_of_choices) + " natural language questions that could be answered by the query given under [QUERY]. Do not use any of the table or column names in the sql query to create the questions. List the questions like this: " + '\n' + "1. This is the first question" + '\n' + "2. This is the second question" + '\n' + "3. This is the third question" + '\n\n' + "[QUERY]" + '\n' + sampled_query_replaced + '\n\n' + "[ABBREVIATIONS] This is a table of all the abbreviations you might need, use this when you see a table or column name to understand the word(s) you will need to use" + '\n\n' + table_string
                chat = [
                    {"role": "user", "content": prompt}
                ]
                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                # tokenize the text
                input_tokens = tokenizer(chat, return_tensors="pt").to(device)
                # generate output tokens
                output = model.generate(**input_tokens, 
                                        max_new_tokens=MAX_NEW_TOKENS)
                # decode output tokens into text
                response = tokenizer.batch_decode(output)
                
                # Get answers from response
                answers = response[0].split("<|end_of_role|>")[2].replace("<|end_of_text|>", "").split("\n")
                output_path = Path(args.output_folder) / f'{idx}_{round_idx}.txt'
                os.makedirs(output_path.parent, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(prompt)
                    f.write('\nOriginal Query:\n')
                    f.write(sampled_query)
                    f.write('\nGranite choices:\n')
                    f.write('\n'.join(answers))

                round_idx += 1

            except ValueError as e:
                print(f'Exception:{e}')
                fail_to_sample += 1

def find_table_name_in_sample(sample):
    parts = sample.split()
    try:
        from_index = parts.index("FROM") + 1
        as_index = parts.index("AS")
        
        table_name = " ".join(parts[from_index:as_index])
    except ValueError:
        print("Required keywords 'FROM' or 'AS' not found in the query.")
    return table_name

def table_name_lookup(args, table):
    connection = sqlite3.connect(args.db_path)
    cursor = connection.cursor()
    schema_with_descriptions = defaultdict(list)
    result = cursor.execute("SELECT * FROM column_label_lookup WHERE Table_name='" + table + "';").fetchall()

    schema_with_descriptions = {}

    for table, column, column_description in result:
        if table not in schema_with_descriptions:
            schema_with_descriptions[table] = []
        schema_with_descriptions[table].append({column: column_description})

    table_string = "Table_name\tColumn_name\tColumn_label\n"
    table_string += "-" * 40 + "\n"  # Separator line

    for table, columns in schema_with_descriptions.items():
        for column_entry in columns:
            for column, column_label in column_entry.items():
                table_string += f"{table}\t{column}\t{column_label}\n"
    return table_string


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