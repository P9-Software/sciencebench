pip install pandas spacy nltk psycopg2-binary torch transformers matplotlib accelerate openai && python3 -m spacy download en_core_web_sm
python3 src/run.py --data_path data/TrialBench --output_folder data/TrialBench/generative/generated
