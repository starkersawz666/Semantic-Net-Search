from flask import Flask, request
import json
import math
import yaml
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
from operator import itemgetter
from itertools import groupby
import dashscope

import extract

app = Flask(__name__)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def load_config():
    file_path = './config.yaml'
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

dashscope.api_key = config['dashscope_api_key']

def generate_schema(config):
    schema = []
    for item in config['basic_filters']:
        schema.append((item, 'bool'))
    for item in config['statistics']:
        for key, value in item.items():
            schema.append((key, value))
    return schema

schema = generate_schema(config)

def load_resources():
    # Download and load word2vec model, and check update if exists
    if config['nltk_downloaded'] == False:
        for resource in config['nltk_resources']:
            nltk.download(resource)
        config['nltk_downloaded'] = True
        with open('./config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    word2vec_model = api.load(config['word2vec_model']['type'])
    ann_model = AnnoyIndex(config['ann_model']['dimensions'], 'angular')
    ann_model.load(config['ann_model']['path'])
    return word2vec_model, ann_model

word2vec_model, ann_model = load_resources()

# Generate statistics options from configuration
def generate_statistics(config):
    statistics = config['statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

def generate_advanced_statistics(config):
    statistics = config['advanced_statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

statistics_options = generate_statistics(config)
statistics_options.sort()

advanced_statistics_option = generate_advanced_statistics(config)
advanced_statistics_option.sort()

def text2vec(text):
    tokens = preprocess_text(text)
    return sum(word2vec_model.get_vector(token) for token in tokens if token in word2vec_model.key_to_index)

def meets_filters(index, basic_filters, advanced_filters, df):
    row = df.loc[df['index'] == index].iloc[0]
    for key, value in basic_filters.items():
        if value == True:
            if row[key] != True:
                return False
        if value == False:
            if row[key] != False:
                return False
    for filter_ in advanced_filters:
        name = filter_[0]
        min_value = filter_[1]
        max_value = filter_[2]
        if min_value == None or min_value == '':
            min_value = -math.inf
        if max_value == None or max_value == '':
            max_value = math.inf
        accept_missing = True
        if row[name] > max_value or row[name] < min_value or row[name] == None and not accept_missing or pd.isna(row[name]) and not accept_missing:
            return False
    return True

def searching(text_input, basic_filters, advanced_filters, return_num, df):
    vector_input = text2vec(text_input)
    if type(vector_input) != np.ndarray or len(vector_input) != config['ann_model']['dimensions']:
        return []
    search_results = []
    n_neighbors = return_num
    while n_neighbors < df.shape[0]:
        if len(search_results) >= return_num:
            break
        if len(search_results) == 0:
            n_neighbors = n_neighbors * 2
        else:
            n_neighbors = math.ceil(n_neighbors / len(search_results) * return_num)
        indexes = ann_model.get_nns_by_vector(vector_input, n_neighbors)
        for index in indexes:
            if index not in search_results and meets_filters(index, basic_filters, advanced_filters, df):
                search_results.append(index)
                if len(search_results) >= return_num:
                    break
    return search_results

def extract_filters(text, schema, max_trials=3):
    cnt_trials = 0
    while cnt_trials < max_trials:
        cnt_trials += 1
        try:
            response = extract.statistics_query(schema, text)
            if response.startswith('Error:'):
                pass
            conditions = extract.extract_conditions(response)
            return conditions
        except:
            if cnt_trials == max_trials:
                return None
            continue

def extract_description(text, max_trials=3):
    cnt_trials = 0
    while cnt_trials < max_trials:
        cnt_trials += 1
        try:
            response = extract.description_query(text)
            if response.startswith('Error:'):
                pass
            return response
        except:
            if cnt_trials == max_trials:
                return None
            continue

def perform_search(search_text, result_count, df):
    conditions = extract_filters(search_text, schema)
    extracted_desc = extract_description(search_text)
    basic_filters = {}
    advanced_filters = []
    if conditions is not None:
        conditions.sort(key = itemgetter(0))
        grouped_conditions = {k: list(g) for k, g in groupby(conditions, key=itemgetter(0))}
        for key, group in grouped_conditions.items():
            if key in statistics_options or key in advanced_statistics_option:
                upper_bound = None
                lower_bound = None
                for item in group:
                    if item[1] == '<' or item[1] == '<=':
                        if upper_bound is None or float(item[2]) < upper_bound:
                            upper_bound = float(item[2])
                    elif item[1] == '>' or item[1] == '>=':
                        if lower_bound is None or float(item[2]) > lower_bound:
                            lower_bound = float(item[2])
                    elif item[1] == '=':
                        upper_bound = lower_bound = float(item[2])
                advanced_filters.append([key, lower_bound, upper_bound])
            else:
                if group[0][2] == 'false':
                    basic_filters[key] = False
                elif group[0][2] == 'true':
                    basic_filters[key] = True
    print(extracted_desc)
    print(basic_filters)
    print(advanced_filters)
    search_results = searching(extracted_desc, basic_filters, advanced_filters, result_count, df)
    default_columns = ['name', 'link', 'description', 'size', 'volume']
    show_columns = default_columns.copy()
    for key, value in basic_filters.items():
        if key not in default_columns:
            show_columns.append(key)
    for filter_ in advanced_filters:
        name = filter_[0]
        if name not in default_columns:
            show_columns.append(name)
    return_table = pd.DataFrame(columns=show_columns)
    for index in search_results:
        row = df.loc[df['index'] == index].iloc[0]
        new_row = [row['name'], row['link'], row['description'], row['size'], row['volume']]
        for key, value in basic_filters.items():
            if key not in default_columns:
                new_row.append(row[key])
        for filter_ in advanced_filters:
            name = filter_[0]
            if name not in default_columns:
                new_row.append(row[name])
        return_table.loc[len(return_table)] = new_row
    return return_table

def load_data():
    file_path = config['datasets']['with_vectors']
    df = pd.read_csv(file_path)
    return df

df = load_data()

@app.route('/search', methods=['GET'])
def search():
    # Obtain search parameters from request
    text = request.args.get('query', default='', type=str)
    max_results = request.args.get('max_results', default=5, type=int)

    # Searches for results
    results = perform_search(text, max_results, df)

    # Return search results as JSON
    return results.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=False)
