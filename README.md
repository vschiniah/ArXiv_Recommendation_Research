# User-item fairness tradeoffs in recommendations

## Overview
This repository contains the source code for a research project focused on understanding user-item fairness tradeoffs in recommendations. This repo contains both the theoretical setup and the empirical setup. 


## ArXiv Recommendation Engine 
### Repository Structure
- **get_authors_papers.py**: Script to fetch papers based on author names from the ArXiv API.
- **get_paper_citations.py**: Script to collect citation data for papers.
- **get_paper_details.py**: Script to retrieve detailed information about each paper.
- **get_references_citations.py**: Script to collect references and citations from the papers.
- **model_evaluation.py**: Contains the code to evaluate the recommendation model.
- **stopwords.txt**: Text file containing stopwords used in text processing.
- **tfidf_authors.py**: Script to compute TF-IDF scores for authors in the dataset.
- **utils.py**: Utility functions used across different scripts.

## Installation

### ArXiv Dataset
The original dataset was sourced from the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).


### Prerequisites
- Python 3.8 or above
- Semantic Scholar API Key

### Dependencies
Install the required Python libraries with pip:
```bash
pip install -r requirements.txt
