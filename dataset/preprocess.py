import json
import sqlite3
from pathlib import Path
from pprint import pformat

def system_prompt():
    return """You are a helpful assistant that generates SQL queries based on the provided database schema and user questions."""

def user_prompt(table_info : str, question : str):
    return f"""For the SQL database with the following schema:
{table_info}
What should be the query for the question: {question}"""

def preprocess_json(input_file, table_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(table_file, 'r') as f:
        table_data = json.load(f)

    error_count = 0

    data_dump = []

    for data_point in data:
        try:
            data_base = data_point['db_id']
            con = sqlite3.connect(Path(f'./spider_data/test_database/{data_base}/{data_base}.sqlite'))
            cursor = con.cursor()
            list_rows = []
            
            for row in cursor.execute(data_point['query']):
                list_rows.append(row)
            data_point['query_result'] = list_rows
            con.close()

            for table in table_data:
                if table['db_id'] == data_base:
                    data_point['table_info'] = table
                    break
            else:
                raise ValueError(f"Table information for database {data_base} not found.")
            
            data_point['prompt'] = [
                {'role': 'system', 'content': system_prompt()},
                {'role': 'user', 'content': user_prompt(pformat(data_point['table_info']), data_point['question'])}
            ]

            data_dump.append(data_point)
        except Exception as e:
            error_count += 1


    with open(output_file, 'w') as f:
        json.dump(data_dump, f, indent=4)

    print(f"Preprocessing completed. Total data points: {len(data_dump)}, Errors: {error_count}")


if __name__ == "__main__":
    preprocess_json('./spider_data/test.json', './spider_data/test_tables.json', './spider_data/preprocessed/preprocessed_test_spider.json')