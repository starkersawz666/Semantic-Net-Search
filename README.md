# SemanticNetSearch

## Introduction
SemanticNetSearch is a tool designed to enhance the efficiency and accuracy of network data searches using large language models and natural language processing technologies. This project integrates automated text processing, SQL generation, and an interactive web interface to assist users in quickly locating and analyzing network data.

## Key Features
- **Text Preprocessing**: Analyzes and preprocesses input text using NLTK and SpaCy.
- **Vectorized Search**: Implements efficient text search capabilities with word2vec and AnnoyIndex for quick nearest neighbor retrieval.
- **SQL Statement Generation**: Generates SQL queries from natural language inputs using tailored prompts and LLM responses.
- **Dynamic Web Interface**: Provides an interactive web interface built with Streamlit, allowing users to input search queries and receive results in real-time.
- **Extensibility**: Allows users to submit datasets to contribute to the database with automatical schema matching and data preprocessing.

## Quick Start
### Installation
```bash
git clone https://github.com/starkersawz666/Semantic-Net-Search.git
cd Semantic-Net-Search
pip install -r requirements.txt
```
### Running the Web Interface
To start the web interface, run:
```bash
streamlit run main.py
```

## Configuration
The project uses a YAML file for configuration settings, including the model configurations, api keys, and database settings. To modify these settings, edit the `config/config.yaml` file accordingly.