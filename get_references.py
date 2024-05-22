import os
import requests
import pandas as pd

BASE_URL = 'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references'
api_key = os.environ.get('API_KEY')

HEADERS = {
    'x-api-key': api_key,
    'Content-Type': 'application/json'
}


def fetch_references(paper_id, paper_title, references):
    """
    Fetches references for a given paper and title.
    :param paper_id: paper_id
    :param paper_title: paper_title
    :param references: Save in references dict
    :return:
    """
    url = BASE_URL.format(paper_id=paper_id)
    response = requests.get(url, headers=HEADERS, params={'fields': 'paperId,title,year,corpusId'})
    if response.status_code == 200:
        data = response.json().get('data', [])
        flattened_data = [entry['citedPaper'] for entry in data if
                          'citedPaper' in entry and entry['citedPaper'] is not None]

        return flattened_data
    else:
        print(f"Failed to fetch details for paper: {paper_id}")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return []


papers_df = pd.read_csv('./Data/test_data.csv')

papers_with_references = {}
references = []
print(len(papers_df))
# Loop through paper IDs and fetch references
for index, row in papers_df.iterrows():
    paper_id = f'{row["s2PaperId"]}'
    paper_title = row['title']
    corpus_id = row['corpusId']
    references = fetch_references(paper_id, paper_title, references)
    papers_with_references[(paper_id, corpus_id, paper_title)] = references

rows = []

for (paper_id, corpus_id, paper_title), references in papers_with_references.items():
    if not references:  # Check if references list is empty
        # Create a row with empty reference information
        row = {
            'Source Paper ID': paper_id,
            'Source Paper Title': paper_title,
            'Source Paper Corpus ID': corpus_id,
            'Reference Paper ID': '',
            'Reference Title': '',
            'Reference Year': '',
            'Reference Authors': ''
        }
        rows.append(row)
    else:
        for ref in references:
            # If ref is a dictionary and not empty
            if ref and isinstance(ref, dict):
                row = {
                    'Source Paper ID': paper_id,
                    'Source Paper Title': paper_title,
                    'Source Paper Corpus ID': corpus_id,
                    'Reference Paper ID': ref.get('paperId', ''),
                    'Reference Paper Corpus ID': ref.get('corpusId', ''),
                    'Reference Title': ref.get('title', ''),
                    'Reference Year': ref.get('year', '')
                }
            else:
                # Create a row with empty reference information if ref is empty
                row = {
                    'Source Paper ID': paper_id,
                    'Source Paper Title': paper_title,
                    'Source Paper Corpus ID': corpus_id,
                    'Reference Paper ID': '',
                    'Reference Paper Corpus ID': '',
                    'Reference Title': '',
                    'Reference Year': '',
                    'Reference Authors': ''
                }
            rows.append(row)


pulled_references = pd.DataFrame(rows)
pulled_references.to_csv('pulled_references.csv')



