import os
import pandas as pd
import requests
from tqdm import tqdm


def search_paper_by_title(title, url, headers):
    """
    Fallback search by paper title.
    """
    params = {'query': title, 'fields': 'paperId,corpusId,title,year,authors'}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json().get('data', [])
        if data:
            return data[0]
    return None


def get_papers(recommendations):
    """
    Get paper information from semantic scholar api
    :param recommendations:
    :return:
    """
    BASE_URL = 'https://api.semanticscholar.org/graph/v1/paper/batch'
    SEARCH_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
    api_key = os.environ.get('API_KEY')

    HEADERS = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }

    # Initialize list to store paper data
    lm_papers = []
    lm_arxiv_ids = [f'arXiv:{id}' for id in recommendations['id']]
    lm_arxiv_title = [f'title:{id}' for id in recommendations['title']]
    for i in tqdm(range(0, len(lm_arxiv_ids), 500)):
        chunk = lm_arxiv_ids[i:i + 500]
        response = requests.post(BASE_URL, headers=HEADERS, json={"ids": chunk},
                                 params={'fields': 'paperId,corpusId,title,year,authors'})
        if response.status_code == 200:
            papers = response.json()
            for ind, paper in enumerate(papers):
                if paper:
                    paper['id'] = paper['paperId'].split(':')
                    lm_papers.append(paper)
                else:
                    title_search_result = search_paper_by_title(
                        url=SEARCH_URL,
                        headers=HEADERS,
                        title=lm_arxiv_title[ind].split(':')[1])
                    if title_search_result:
                        lm_papers.append(title_search_result)
                    else:
                        print(f"Failed to fetch details for paper ID: {lm_arxiv_title[ind]}")
        else:
            print("Status Code:", response.status_code)
            print("Response:", response.text)

    # Convert the list of papers into a DataFrame
    lm_s2_df = pd.DataFrame.from_records(lm_papers)
    lm_s2_df = lm_s2_df.rename(columns={'paperId': 's2PaperId'}).set_index('id').reset_index()
    combined = pd.merge(recommendations, lm_s2_df, on='title', how='left')
    return combined


data_df = pd.read_csv("/Data/arxiv_data.csv")
data_s2 = get_papers(data_df)
data_s2 = data_s2.dropna(subset=['corpusId'])
data_s2 = data_s2.drop_duplicates(subset=['title'])
data_s2.to_csv('test2.csv')
