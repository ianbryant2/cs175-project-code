import json
import sqlite3
from pathlib import Path
from pprint import pformat
import argparse

def system_prompt():
    return """You are a Text-to-SQL generator.

Rules:
1. Analyze the Schema constraints and question in <think> tags.
2.  Output a single, executable query in <sql> tags."""

def user_prompt(table_info : str, question : str):
    return f"""For the SQL database with the following schema:
{table_info}
Create a query for the question: {question}"""

def preprocess_json(input_file : Path, table_file : Path, dataset : str) -> list:
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(table_file, 'r') as f:
        table_data = json.load(f)

    error_count = 0

    data_dump = []

    if dataset.lower() == 'test':
        ds_folder = 'test_database'
    else:
        ds_folder = 'database'

    for data_point in data:
        try:
            data_base = data_point['db_id']
            con = sqlite3.connect(Path(f'./spider_data/{ds_folder}/{data_base}/{data_base}.sqlite'))
            cursor = con.cursor()
            list_rows = []
            
            for row in cursor.execute(data_point['query']):
                list_rows.append(str(row))
            data_point['query_result'] = list_rows
            data_point['query_result_columns'] = list([desc[0].lower() for desc in cursor.description])
            con.close()

            for table in table_data:
                if table['db_id'] == data_base:
                    data_point['table_info'] = pformat(table)
                    break
            else:
                raise ValueError(f"Table information for database {data_base} not found.")
            
            data_point['prompt'] = [
                {'role': 'system', 'content': system_prompt()},
                {'role': 'user', 'content': user_prompt(data_point['table_info'], data_point['question'])}
            ]

            del data_point['sql']

            data_dump.append(data_point)
        except Exception as e:
            error_count += 1

    print(f"Preprocessing {input_file} completed. Total data points: {len(data_dump)}, Errors: {error_count}")

    return data_dump

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to processs (train or test)', default=None)
    parser.add_argument('-o', '--output', type=str, help='File name to save the dataset to', default=None)
    parser.add_argument('-a', '--additional', action='store_true', help='If to include the additional datapoints in dev and others into the training data')
    args = parser.parse_args()

    if args.dataset.lower() == 'train':
        train = preprocess_json(Path('spider_data/train_spider.json'), Path('spider_data/tables.json'), 'train')

        if args.additional:
            dev = preprocess_json(Path('spider_data/dev.json'), Path('spider_data/tables.json'), 'train')
            others = preprocess_json(Path('spider_data/train_others.json'), Path('spider_data/tables.json'), 'train')
            train.extend(dev)
            train.extend(others)

        with open(Path('spider_data/preprocessed') / args.output, 'w') as f:
            json.dump(train, f, indent=4)
    else:
        test = preprocess_json(Path('spider_data/test.json'), Path('spider_data/test_tables.json'), 'test')
        with open(Path('spider_data/preprocessed') / args.output, 'w') as f:
            json.dump(test, f, indent=4)