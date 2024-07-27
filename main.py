_ = '''
Code Part 1: Loading and Searching Functions

preprocess_text(): tokenizing given text and removing stop words
text2vec(): computing the vector representation of given text using word2vec model
meets_filters(): checking if a row meets the basic filters and advanced filters
searching(): searching for similar texts using AnnoyIndex and filtering results based on basic and advanced filters
'''

import gensim.downloader as api
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
from annoy import AnnoyIndex
import yaml
import numpy as np
import math
import sys
from itertools import groupby
from streamlit_modal import Modal
from operator import itemgetter
import dashscope
from dashscope import Generation
import spacy
import re
import datetime

from styles import css_style
from utils import randkey
from utils import schemas
from utils import extract
from utils import highlight

# Configure page to use a wide layout
st.set_page_config(layout="wide")

# Load configuration from YAML file
@st.cache_data()
def load_config():
    print('Config loading started')
    file_path = './config/config.yaml'
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    print('Config loaded')
    return config

config = load_config()

# Load required models for text processing
@st.cache_resource()
def load_resources():
    print('Model loading started')
    # Download and load word2vec model, and check update if exists
    if config['nltk_downloaded'] == False:
        print('ntlk models downloading')
        for resource in config['nltk_resources']:
            nltk.download(resource)
        config['nltk_downloaded'] = True
        with open('./config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print('word2vec models downloading')
    word2vec_model = api.load(config['word2vec_model']['type'])
    ann_model = AnnoyIndex(config['ann_model']['dimensions'], 'angular')
    ann_model.load(config['ann_model']['path'])
    if config['spacy_downloaded'] == False:
        print('spacy model downloading')
        for resource in config['spacy_nlp_models']:
            spacy.cli.download(resource)
        config['spacy_downloaded'] = True
        with open('./config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    spacy_nlp = spacy.load('en_core_web_sm')
    print('Model loaded')
    return word2vec_model, ann_model, spacy_nlp

word2vec_model, ann_model, spacy_nlp = load_resources()

# Load data from dataset
@st.cache_data()
def load_data():
    print('Data loading started')
    file_path = config['datasets']['with_vectors']
    df = pd.read_csv(file_path)
    print('Data loaded')
    return df

df = load_data()

# Text preprocessing function to tokenize and remove stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Convert text to vector using word2vec model
def text2vec(text):
    tokens = preprocess_text(text)
    return sum(word2vec_model.get_vector(token) for token in tokens if token in word2vec_model.key_to_index)

# Check if a row meets specified filters
def meets_filters(index, basic_filter_keys, advanced_filters, df):
    row = df.loc[df['index'] == index].iloc[0]
    for key in basic_filter_keys:
        if st.session_state[key] == 'True':
            if row[key] != True:
                return False
        if st.session_state[key] == 'False':
            if row[key] != False:
                return False
    for filter_ in advanced_filters:
        name = filter_['category']
        min_value = filter_['min_value']
        max_value = filter_['max_value']
        accept_missing = filter_['accept_missing']
        if min_value == None or min_value == '':
            min_value = -math.inf
        if max_value == None or max_value == '':
            max_value = math.inf
        if row[name] > max_value or row[name] < min_value or row[name] == None and not accept_missing or pd.isna(row[name]) and not accept_missing:
            return False
    return True

# Perform search based on filters
def searching(text_input, basic_filter_keys, advanced_filters, return_num, df):
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
            if index not in search_results and meets_filters(index, basic_filter_keys, advanced_filters, df):
                search_results.append(index)
                if len(search_results) >= return_num:
                    break
    return search_results

statistics_options = schemas.generate_statistics(config)
statistics_options.sort()
advanced_statistics_option = schemas.generate_advanced_statistics(config)
advanced_statistics_option.sort()
schema = schemas.generate_schema(config)

dashscope.api_key = config['dashscope_api_key']

# Extract filters from text using dashscope
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
            print(str(sys.exc_info()[0]))
            error_modal = Modal(
                "Error", 
                key="error-modal-" + str(cnt_trials),
                padding=15,
                max_width=500
            )
            if error_modal.is_open():
                with error_modal.container():
                    st.write("Unexpected error:" + str(sys.exc_info()[0]))
            if cnt_trials == max_trials:
                error_modal.open()
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
            print(str(sys.exc_info()[0]))
            error_modal = Modal(
                "Error", 
                key="error-modal-" + str(cnt_trials),
                padding=15,
                max_width=500
            )
            if error_modal.is_open():
                with error_modal.container():
                    st.write("Unexpected error:" + str(sys.exc_info()[0]))
            if cnt_trials == max_trials:
                error_modal.open()
                return None
            continue


_ = '''
Part 2: Streamlit Interface

perform_search(): dealing with searching results and generate a table for display
'''

css_style.css_style()

# Initialize session state for filters
if 'filters' not in st.session_state:
    st.session_state.filters = []

# Add a default filter to the session state
def add_default_filter():
    st.session_state.filters.append({'category': 'size', 'min_value': None, 'max_value': None, 'accept_missing': True})

def add_advanced_default_filter():
    st.session_state.filters.append({'category': 'left_size', 'min_value': None, 'max_value': None, 'accept_missing': True})

# Add a filter with specified parameters
def add_filter(category, min_value, max_value, accept_missing):
    i = len(st.session_state.filters)
    st.session_state.filters.append({'category': '', 'min_value': 0, 'max_value': 100, 'accept_missing': True})
    st.session_state.filters[i]['category'] = category
    st.session_state.filters[i]['min_value'] = min_value
    st.session_state.filters[i]['max_value'] = max_value
    st.session_state.filters[i]['accept_missing'] = accept_missing

# Delete a filter / filters from session state
def delete_filter(index):
    del st.session_state.filters[index]

def delete_all_filters():
    st.session_state.filters = []

# Perform search operation using text input and filters
def perform_search(search_text, basic_filter_keys, advanced_filters, result_count, df):
    search_results = searching(search_text, basic_filter_keys, advanced_filters, result_count, df)
    default_columns = ['name', 'link', 'description', 'size', 'volume']
    show_columns = default_columns.copy()
    for key in basic_filter_keys:
        if st.session_state[key] != 'None' and key not in default_columns:
            show_columns.append(key)
    for filter_ in advanced_filters:
        name = filter_['category']
        if name not in default_columns:
            show_columns.append(name)
    return_table = pd.DataFrame(columns=show_columns)
    for index in search_results:
        row = df.loc[df['index'] == index].iloc[0]
        new_row = [row['name'], row['link'], row['description'], row['size'], row['volume']]
        for key in basic_filter_keys:
            if st.session_state[key] != 'None' and key not in default_columns:
                new_row.append(row[key])
        for filter_ in advanced_filters:
            name = filter_['category']
            if name not in default_columns:
                new_row.append(row[name])
        return_table.loc[len(return_table)] = new_row
    return return_table


def submit_data(user_choices, original_df, main_attrs):
    result_df = pd.DataFrame()
    for new_col, original_col in user_choices.items():
        if original_col is None:
            continue
        if original_col in main_attrs:
            result_df[original_col] = original_df[new_col].copy()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    rand_str = randkey.random_string(10)
    file_name = f"data_{timestamp}_{rand_str}.csv"
    result_df.to_csv("./datasets/user_submission/" + file_name, index=False)

_ = '''
Part 3: Search Page
'''

# Streamlit page for search interface
def page_search():
    # Basic Filters: True / False / None filters
    basic_options = {}
    for option in config['basic_filters']:
        basic_options[option] = 'None'

    for key in basic_options:
        if key not in st.session_state:
            st.session_state[key] = basic_options[key]

    # Streamlit page
    st.markdown("<h1>SemanticNetSearch Interface</h1>", unsafe_allow_html=True)
    # st.title("Network Search Interface")
    input_column, button_column = st.columns([4, 1])

    if 'search_text' not in st.session_state:
        st.session_state['search_text'] = ''
    
    with input_column:
        search_text = st.text_area("Search Text", value = st.session_state['search_text'])

    with button_column:
        st.markdown('<div class="extract-button">', unsafe_allow_html=True)
        # Add some blank lines
        st.markdown('<br><br>', unsafe_allow_html=True)
        if st.button("Extract Filters"):
            # Condition Extraction
            conditions = extract_filters(search_text, schema)
            if conditions is not None:
                delete_all_filters()
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
                        add_filter(key, lower_bound, upper_bound, True)
                    else:
                        if group[0][2] == 'false':
                            st.session_state[key] = 'False'
                        elif group[0][2] == 'true':
                            st.session_state[key] = 'True'
            else:
                print("No conditions extracted")
            
            # Description Extraction
            extracted_desc = extract_description(search_text)
            if extracted_desc is not None:
                st.session_state['search_text'] = extracted_desc
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    


    # Display the basic filters using an expander
    basic_filters = st.expander("Basic Filters")
    with st.expander("Basic Filters"):
        for key in basic_options.keys():
            options = ('None', 'True', 'False')
            if key not in st.session_state:
                st.session_state[key] = 'None'
            
            # For better display
            label = key.replace("_", " ").title()

            # Test if the filter is applicable
            if key == 'reciprocal' and 'directed' in st.session_state and st.session_state['directed'] == 'False':
                st.session_state[key] = 'None'
            
            if key == 'directed_cycle' and 'loops' in st.session_state and st.session_state['loops'] == 'False':
                st.session_state[key] = 'None'

            if (key == 'positive_weights' or key == 'negative_weights') and 'weighted' in st.session_state and st.session_state['weighted'] == 'False':
                st.session_state[key] = 'None'

            if key == 'directed_cycle' and 'directed' in st.session_state and st.session_state['directed'] == 'False':
                st.session_state[key] = 'None'

            # Display the filter
            selected_option = st.radio(
                label=label,
                options=options,
                index=options.index(st.session_state[key]), 
                format_func=lambda x: 'None' if x == 'None' else ('True' if x == 'True' else 'False'),
                # key=key,
                horizontal=True,
            )

            # For unapplicable filters, disallow user to select them and show the reason
            if key == 'reciprocal' and 'directed' in st.session_state and st.session_state['directed'] == 'False':
                st.markdown("<span class='bold-text'>RECIPROCAL</span> is not applicable if the graph is undirected.", unsafe_allow_html=True)
                # st.write(f"RECIPROCAL is not applicable if the graph is undirected.")
            
            if key == 'directed_cycle' and 'loops' in st.session_state and st.session_state['loops'] == 'False':
                st.markdown("<span class='bold-text'>DIRECTED CYCLE</span> is not applicable if the graph has no loops.", unsafe_allow_html=True)
                # st.write(f"DIRECTED CYCLE is not applicable if the graph has no loops.")

            if (key == 'positive_weights' or key == 'negative_weights') and 'weighted' in st.session_state and st.session_state['weighted'] == 'False':
                st.markdown("<span class='bold-text'>" + key.replace("_", " ").title() + "</span> is not applicable if the graph is unweighted.", unsafe_allow_html=True)
                # st.write(f"{key.replace('_', ' ').title()} is not applicable if the graph is unweighted.")
            
            if key == 'directed_cycle' and 'directed' in st.session_state and st.session_state['directed'] == 'False':
                st.markdown("<span class='bold-text'>DIRECTED CYCLE</span> is not applicable if the graph is undirected.", unsafe_allow_html=True)



    # The number of results needed, input by user
    result_count = st.number_input("# Results Needed", min_value=1, value=5)

    # Advanced Filters: filters for statistical properties of the network
    st.markdown("Statistical Filters", unsafe_allow_html=True)

    # showing the current advanced filters
    for i, filter_ in enumerate(st.session_state.filters):
        cols = st.columns([10.2, 4, 4, 5.3, 2.5, 6.5])
        use_options = statistics_options if filter_['category'] in statistics_options else advanced_statistics_option
        with cols[0]:
            st.session_state.filters[i]['category'] = st.selectbox("Category", use_options, key=f"category_{i}", index = use_options.index(filter_['category']))
        with cols[1]:
            st.session_state.filters[i]['min_value'] = st.number_input("Min", key=f"min_{i}", value = st.session_state.filters[i].get('min_value', 0))
        with cols[2]:
            st.session_state.filters[i]['max_value'] = st.number_input("Max", key=f"max_{i}", value = st.session_state.filters[i].get('max_value', 100))
        with cols[3]:
            current_value = st.session_state.filters[i].get('accept_missing', True)
            st.session_state.filters[i]['accept_missing'] = st.checkbox("Accept Missing Values", value=current_value, key=f"accept_{i}")
        with cols[4]:
            st.button("Delete", on_click=delete_filter, args=(i,), key=f"delete_{i}")

    # Add-advanced-filter button
    statistics_button1, statistics_button2 , _= st.columns([1.3, 2, 5])
    with statistics_button1:
        st.button("Add Statistical Filter", on_click=add_default_filter)
    with statistics_button2:
        st.button("Add Advanced Statistical Filter", on_click=add_advanced_default_filter)

    # Searching button
    if st.button("Search"):
        if not search_text:
            st.warning("Search text cannnot be empty")
        else:
            results = perform_search(
                search_text.replace('network', ''), 
                basic_options.keys(), 
                st.session_state.filters, 
                result_count,
                df
            )
            # Display the searching results
            st.markdown("<div class='search-results'>Search Results:</div>", unsafe_allow_html=True)
            # st.write("Search Results:")
            st.markdown("---")
            
            for index, row in results.iterrows():
                with st.container():
                    st.markdown(f"### <a href='{row['link']}' target='_blank' class='custom-link'>{row['name']}</a>", unsafe_allow_html=True)
                    # st.markdown(f"### [{row['name']}]({row['link']})", unsafe_allow_html = True)
                    description_text = row['description']
                    random_mark = randkey.random_string()
                    highlighted_text = highlight.highlight_texts(search_text.replace('network', ''), description_text, spacy_nlp, random_mark)
                    highlight_start_tag = "<span class='highlight-keyword'>"
                    highlight_end_tag = "</span>"
                    pattern = rf'{re.escape(random_mark)}(.*?){re.escape(random_mark)}'
                    highlighted_text = re.sub(pattern, f"{highlight_start_tag}\\1{highlight_end_tag}", highlighted_text)
                    st.markdown(f"<span class='description'>{highlighted_text}</span>", unsafe_allow_html=True)
                    for col in sorted(results.columns):
                        if col not in ['name', 'link', 'description']:
                            if pd.isna(row[col]):
                                value = 'Unknown'
                            else:
                                value = row[col]
                            st.markdown(f"<span class='bold-text'>{col.replace('_', ' ').title()}: </span>{value}", unsafe_allow_html=True)
                            #st.text(f"{col.replace('_', ' ').title()}: {value}")
                    st.markdown(f"[Read More]({row['link']})", unsafe_allow_html = True)
                    st.markdown("---")


def page_submission():
    st.markdown("<h1>SemanticNetSearch Datasets Submission</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Select a file to upload:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("label", label_visibility='collapsed', type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        df = None
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "application/vnd.ms-excel.sheet.binary.macroEnabled.12":
            df = pd.read_excel(uploaded_file, engine='pyxlsb')
        if df is None:
            st.markdown("<h4>Error: Illegal File. Please try again.</h4>", unsafe_allow_html=True)
        else:
            st.markdown("Automatically matching...", unsafe_allow_html=True)
            sub_attrs = [x for x in df.columns]
            main_attrs = ['name', 'link', 'description', 'category']
            main_attrs.extend(statistics_options)
            main_attrs.extend(advanced_statistics_option)
            for option in config['basic_filters']:
                main_attrs.append(option)
            nlp = spacy.load('en_core_web_md')
            matching = extract.match_schema(main_attrs, sub_attrs, nlp, threshold=0.8, max_saved_attrs=6)
            user_choices = {}
            st.markdown("<h4>Automatical matching result:</h4>", unsafe_allow_html=True)
            if matching is not None:
                main_attrs.sort()
                main_attrs.append(None)
                keys = list(matching.keys())
                num_keys = len(keys)
                num_rows = (num_keys + 2) // 3
                for i in range(num_rows):
                    cols = st.columns(3)
                    for j in range(3):
                        index = 3 * i + j
                        if index < num_keys:
                            key = keys[index]
                            with cols[j]:
                                selected_value = st.selectbox(
                                    f"{key}: ",
                                    options=main_attrs,
                                    index=main_attrs.index(matching[key]) if matching[key] in main_attrs else 0
                                )
                                user_choices[key] = selected_value
                if st.button("Confirm and Submit"):
                    submit_data(user_choices, df, main_attrs)
                    st.success("Your data has been submitted successfully.")

def page_tutorial():
    st.title("User Guide for SemanticNetSearch")

    st.header("Introduction")
    st.write("""
    SemanticNetSearch is an advanced tool designed to empower researchers with semantic search technology to access network datasets. Our platform provides an intuitive web interface for extracting relevant information, searching through extensive datasets, and submitting your data for analysis. Access the application by visiting http://ns2.nilou.top, and no prior installation is required.
    """)

    st.header("Semantic Searching")
    st.subheader("Step 1: Requirements Extraction")
    st.write("""
    - Enter your requirements in natural language in the text area, and click 'Extract Filters' to extract the filters from the text.
    """)
    st.write("You can try the following example: An undirected network about the relationship between users on social media, with more than 10000 nodes, 100000 edges, a max degree between 500 and 2000, and an average degree smaller than 50.")

    video_col1 = st.columns([1, 1])
    with video_col1[0]:
        st.video('./videos/Extraction.mp4', format='video/mp4', start_time=0, loop=True)

    st.subheader("Step 2: Filters Adjustment")

    st.write("""
    - After the filters have been extracted from your input, they will appear in a manageable list below. This list allows you to review and adjust the filters to refine your search criteria.
    """)
    st.write("You can try to cancel all the 'accept missing' checkboxes.")

    video_col2 = st.columns([1, 1])
    with video_col2[0]:
        st.video('./videos/Adjustment.mp4', format='video/mp4', start_time=0, loop=True)

    st.subheader("Step 3: Searching")

    st.write("""
    - Click the Search button below to perform the search. The system will return a list of relevant datasets that match your criteria. You can click on the dataset name to view its detailed information.
    """)

    st.header("Data Submission")
    st.write("""
    - **Uploading Files**: Click on 'Upload File' button and select the file you wish to submit. Supported formats include CSV and Excel.
    - **Auto-matching**: The system will automatically try to match your data columns to our database. You can review and adjust these mappings before final submission.
    - **Submission**: Once satisfied with the mappings, click 'Submit' to finalize your data submission.
    """)
    st.write("You can try data submission function on our sample dataset:")
    sample_csv_path = "./datasets/sample_submission.csv"
    with open(sample_csv_path, "rb") as file:
        st.download_button(
            label="Download Sample CSV File",
            data=file,
            file_name="sample_submission.csv",
            mime='text/csv'
        )

    st.header("FAQ")
    st.write("""
    - **Q**: Are the search results reproducible?
    - **A**: The "Filters Extraction" process has a certain degree of randomness due to the use of LLM, but the search results are reproducible when the search text and conditions are exactly the same.
    """)

    st.header("Getting Help")
    st.write("""
    If you encounter any issues or have questions, please contact our support team at zixinwei1@link.cuhk.edu.cn.
    """)

def page_about():
    pass

# Handling different pages in the sidebar
st.sidebar.markdown("""
<style>
.sidebar-title {
    font-size:35px !important;
    font-weight:bold !important;
}
</style>
<div class='sidebar-title'>Navigation</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Choose a page:", ("Search", "Datasets Submission", "Tutorial", "About"))

if page == "Search":
    page_search()
elif page == "Datasets Submission":
    page_submission()
elif page == "Tutorial":
    page_tutorial()
elif page == "About":
    page_about()

