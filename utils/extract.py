_ = """
This file contains functions for generating SQL statements and extracting information from the user's input / LLMs' response.
"""

import dashscope
from dashscope import Generation
from http import HTTPStatus
import re
import yaml
import json

# Load configuration settings from YAML file
def load_api():
    print('Config loading started')
    file_path = '../config/config.yaml'
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    print('extract.py: Config loaded')
    return config['dashscope_api_key']

# Send a request to the Dashscope Generation API and get a response
def get_response(messages):
    response = Generation.call(
        "qwen-plus",
        messages=messages,
        result_format="message",
    )
    return response

# Create an AI prompt for generating statistics based on a schema
def statistics_prompt(schema):
    prompt = "You are an AI database query assistant. When a user inputs their query about a network, you are to output PostgreSQL statements based on the given table structures and nothing else. Here is the structure of the table you will interact with: \n"
    prompt += "CREATE TABLE `network` (\n"
    for item in schema:
        name = item[0]
        data_type = item[1]
        if data_type == "float":
            prompt += "`" + name + "` DOUBLE,\n"
        elif data_type == "int":
            prompt += "`" + name + "` INT,\n"
        elif data_type == "bool":
            prompt += "`" + name + "` BOOLEAN,\n"
    prompt += ");"
    prompt += "User Input: 'Describe your query or the information you need from the database.'\n"
    prompt += "Your Task: Generate a SQL query statement that accurately addresses the user's request based on the provided table structure. Your response should consist solely of the SQL statement, without any additional text or explanation. If the user input contains nothing about the given schema, return an error message."
    return prompt

# Process a user query and generate SQL statements through AI
def statistics_query(schema, text):
    messages = [{"role": "system", "content": statistics_prompt(schema)}]
    messages.append({"role": "user", "content": text})
    response = get_response(messages)
    if response.status_code == HTTPStatus.OK:
        assistant_output = response.output.choices[0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_output})
        return assistant_output
    else:
        return ('Error: Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

# Prompt AI for extracting descriptive sentences about a graph
def description_prompt():
    prompt = ("You are an AI assistant specialized in processing network search queries. You have received a user's original search input, along with extracted Properties and Statistics. Your task is to remove all content related to Properties and Statistics from the original input, leaving only the Content-related part."
                "Example User Input:"
                "Original Search Input: I am looking for a non-directed network about the relationship between users on social media with more than 10,000 nodes, a maximum degree between 50 and 100, and an average degree of 5 to 10."
                "Extracted Properties: Non-directed"
                "Extracted Statistics: More than 10,000 nodes, maximum degree between 50 and 100, average degree between 5 and 10."
                "Your Task: Identify and remove all mentions of the extracted Properties and Statistics. Retain the description related to the network Content, ensuring the final output includes only this aspect."
                "Output: Only the Content description without any number or property, such as: “the relationship between users on social media."
                "Do not return other messages other than the Output content itself.")
    return prompt

# Query AI for a description based on the prompt and process response
def description_query(text):
    messages = [{"role": "system", "content": description_prompt()}]
    messages.append({"role": "user", "content": text})
    response = get_response(messages)
    if response.status_code == HTTPStatus.OK:
        assistant_output = response.output.choices[0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_output})
        # print(assistant_output)
        return assistant_output
    else:
        return ('Error: Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

# Extract SQL conditions from a given SQL query, only for SELECT queries
def extract_conditions(sql_query):
    start = sql_query.lower().find('where')
    if start == -1:
        return []
    end = sql_query.lower().find('order by')
    if end == -1:
        end = sql_query.lower().find('group by')
    if end == -1:
        end = sql_query.lower().find(';')
    if end == -1:
        end = len(sql_query)
    conditions_text = sql_query[start + len('where'):end].strip().lower()
    while True:
        between_index = conditions_text.find('between')
        if between_index == -1:
            break
        and_index = conditions_text.find('and', between_index)
        tmp = conditions_text[:and_index] + 'ad' + conditions_text[and_index+3:]
        conditions_text = tmp[:between_index] + 'bw' + tmp[between_index+7:] 
    conditions_text = re.sub(r'\s+', ' ', conditions_text.replace('\n', ' '))
    raw_conditions = conditions_text.split(' and ')
    conditions = []
    for condition in raw_conditions:
        if 'bw' in condition.lower():
            parts = condition.split()
            column = parts[0]
            lower_bound = parts[2]
            upper_bound = parts[4]
            conditions.append((column, '>=', lower_bound))
            conditions.append((column, '<=', upper_bound))
        else:
            parts = condition.split()
            column = parts[0]
            operator = parts[1]
            value = parts[2]
            conditions.append((column, operator, value))
    return conditions

# Check if a given string is a valid SQL statement
def is_sql_statement(sql_str):
    pattern = re.compile(r"^\s*(SELECT|UPDATE|DELETE|INSERT INTO|CREATE TABLE|ALTER TABLE|DROP TABLE|TRUNCATE TABLE)\s+.*[;]?\s*$", re.IGNORECASE)
    if pattern.match(sql_str):
        return True
    else:
        return False
    
def extraction_prompt(input_type: str, output_type: str, is_list: bool, input_description: str, output_description: str, conversion_description=None, additional_reminder=""):
    output_guide = None
    if not is_list:
        output_guide = f"a JSON object that describes an object of type {output_type}"
    else:
        output_guide = f"an array of zero or more JSON objects that describe objects of type {output_type}"
    prompt = f"""
        You are a JSON extractor bot. I would like you to extract {output_guide} from the input of type '{input_type}'.
        The input is :{input_description}. The output should be :{output_description}.
        Keep in mind that the output should be in JSON format.
        {f"The conversion process is described as follows: {conversion_description}." if conversion_description is not None else ""}
        When the input is given by the user, please provide the output in JSON format, and include nothing other than the needed output of type {output_type}{"as a list" if is_list else ""}. {additional_reminder}
    """
    return prompt

def extract_json_from_answer(answer: str):
    start_index = end_index = -1
    if '{' in answer:
        start_index = answer.find("{")
    else:
        start_index = answer.find("[")
    if '}' in answer:
        end_index = answer.rfind("}")
    else:
        end_index = answer.rfind("]")
    if start_index != -1 and end_index != -1:
        answer = answer[start_index:end_index+1].replace("\_", "_")
    return json.loads(re.sub(r'\/\/.*$', '', answer, flags=re.MULTILINE))

def match_schema(main_schema, sub_schema, nlp, threshold=0.85, max_saved_attrs=5):
    return_schema = {}
    for attr in sub_schema:
        # if the key is in the main schema, directly add it to the return schema
        if attr in main_schema:
            return_schema[attr] = attr
        else:
            # save the 5 most similar attributes in the main schema
            similar_attrs = []
            for main_attr in main_schema:
                word1 = nlp(main_attr.replace("_", " "))
                word2 = nlp(attr.replace("_", " "))
                similarity = word1.similarity(word2)
                if len(similar_attrs) < max_saved_attrs:
                    similar_attrs.append((main_attr, similarity))
                else:
                    similar_attrs.sort(key=lambda x: x[1], reverse=True)
                    if similarity > similar_attrs[max_saved_attrs - 1][1]:
                        similar_attrs[max_saved_attrs - 1] = (main_attr, similarity)
            # sort the attrs by similarity
            similar_attrs.sort(key=lambda x: x[1], reverse=True)
            # if the highest similarity is above the threshold, add it to the return schema
            if similar_attrs[0][1] > threshold:
                return_schema[attr] = similar_attrs[0][0]
            # else, test by LLM
            else:
                prompt = extraction_prompt('str', 'bool', False, 'a few relevant words', 'whether one of the words in the input has the same meaning to the word' + attr.replace("_", " "), None, "")
                messages = [{"role": "system", "content": prompt}]
                messages.append({"role": "user", "content": "The word is: " + attr.replace("_", " ") + ", and the possible relevant words are: " + (", ".join(x[0] for x in similar_attrs)).replace("_", " ") + "."})
                response = get_response(messages)
                content = response.output.choices[0]["message"]["content"]
                json_value = extract_json_from_answer(content)
                bool_value = next(iter(json_value.values()))
                if bool_value:
                    prompt = extraction_prompt('str', 'int', False, 'a few relevant words', 'the index of the word (start from 0) among the input that has the same meaning to the word' + attr.replace("_", " "), None, "")
                    messages = [{"role": "system", "content": prompt}]
                    messages.append({"role": "user", "content": "The word is: " + attr.replace("_", " ") + ", and the possible relevant words are: " + (", ".join(similar_attrs[i][0] + '(' + str(i) + ')' for i in range(len(similar_attrs)))).replace("_", " ") + "."})
                    response = get_response(messages)
                    content = response.output.choices[0]["message"]["content"]
                    json_value = extract_json_from_answer(content)
                    int_value = json_value if type(json_value) == int else next(iter(json_value.values()))
                    return_schema[attr] = similar_attrs[int_value][0]
                else:
                    return_schema[attr] = None
    return return_schema