import pandas as pd
import re
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from utils import extract_authors, preprocess_text
from model_eval_functions_v3 import rec_referenced_by_author, rec_cited_author
import time
import concurrent
import concurrent.futures


start_time = time.perf_counter()

# Import dataset
train_data = pd.read_csv('./Data/train_data.csv')


#  Test Data
test_data = pd.read_csv('./Data/all_test_data_s2.csv')

test_data['full_names'] = test_data['authors_parsed'].apply(extract_authors)
# all_full_names = sum(test_data['full_names'], [])
# unique_full_names = set(all_full_names)

with open('./Data/tfidf_matrix_all_train_data.pickle', 'rb') as f:
    tfidf_matrix_train_data = pickle.load(f)

with open('./Data/tfidf_matrix_all_test_data.pickle', 'rb') as r:
    tfidf_matrix_test_data = pickle.load(r)


# TODO: Recommendation function
def recommend_papers(author, train_data, test_data=None, tfidf_matrix_train_data=None, tfidf_matrix_test_data=None):

    author_df = train_data[train_data['authors'].apply(lambda authors: author in authors)]
    trained_on = (author, len(author_df))

    if not author_df.empty:
        # Creating the TF-IDF Vectorizer and computing the cosine similarity matrix
        author_indices = author_df.index.tolist()
        authors_matrix = tfidf_matrix_train_data[author_indices]
        # with open(f'./Data/transformer_batch_mean/batch_pickle_files/{author}_matrix.pickle', 'wb') as fin:
        #     pickle.dump(authors_matrix, fin)
        # print(f'successfully pulled embedding for {author} from train data')
        cosine_sim = cosine_similarity(authors_matrix, tfidf_matrix_test_data)
        avg_cosine_sim = cosine_sim.max(axis=0)
        scores_per_paper = {test_data.iloc[i]['title']: score for i, score in enumerate(avg_cosine_sim)}

        avg_sim_scores = []

        # Iterate over the author's papers in the cosine_sim matrix
        for jdx, score in enumerate(avg_cosine_sim):
            # if jdx not in author_indices:
            avg_sim_scores.append((jdx, score))

        # Sort papers based on similarity scores
        avg_sim_scores = sorted(avg_sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 10 recommendations (excluding author's own papers)
        # top_paper_indices = [i[0] for i in avg_sim_scores[:10]]
        all_paper_indices = [i[0] for i in avg_sim_scores]

        recommended_df = test_data.iloc[all_paper_indices][['id', 'title', 'authors', 'abstract', 'authors_parsed',
                                                            's2PaperId','corpusId', 'year']]
        recommended_df['recommended_to'] = author
        recommended_df['number of papers trained on'] = trained_on[1]
        recommended_df['similarity_score'] = [avg_cosine_sim[idx] for idx in all_paper_indices]
        return recommended_df, scores_per_paper
    else:
        result_df = pd.DataFrame(columns=['id', 'title', 'authors', 'abstract', 'authors_parsed', 's2PaperId', 'corpusId', 'year'])
        return result_df, {'info': f'No data available for {author}'}


def recommend_for_authors(authors_list, train_data, test_data=None,tfidf_matrix_train_data=None, tfidf_matrix_test_data=None):
    scores_data = []
    print('authors_list', authors_list)
    all_recommendations = pd.DataFrame()
    for author in authors_list:
        recommended_papers, scores_per_paper = recommend_papers(author, train_data, test_data,
                                                                tfidf_matrix_train_data, tfidf_matrix_test_data)
        if not recommended_papers.empty:
            all_recommendations = pd.concat([all_recommendations, recommended_papers], ignore_index=True)
            scores_data.append({'author': author, **scores_per_paper})
        else:
            # Handle the case where no papers are recommended
            scores_data.append({'author': author})

    scores_df = pd.DataFrame(scores_data)
    scores_df = scores_df.set_index('author')

    return all_recommendations, scores_df


pulled_references = pd.read_csv('./Data/all_pulled_references_s2.csv')
pulled_citations = pd.read_csv('./Data/all_pulled_citations_s2.csv')
references_df = pd.read_csv('./Data/all_papers_and_authors_s2.csv')
## Removed these for GitHub
authors_list_1 = []
authors_list_2 = []
combined_auth_list = authors_list_1 + authors_list_2
authors_list = list(set(combined_auth_list))


def process_authors_batch(i, authors_batch):
    all_recommendations, scores_df = recommend_for_authors(authors_batch, train_data, test_data,
                                                           tfidf_matrix_train_data,
                                                           tfidf_matrix_test_data)
    all_recommendations.to_csv(f"./Data/new_tfidf_batch_v3_7_19_2024/all_recommendations_authors_method_1000aut_max_{i}.csv")
    scores_df.to_csv(f"./Data/new_tfidf_batch_v3_7_19_2024/average_author_similarity_scores_1000aut_max_{i}.csv")
    print(f'successfully created recommendations {i}', flush=True)

    combined_recommendations = rec_referenced_by_author(all_recommendations, pulled_references, authors_batch,
                                                        references_df)
    print(f'successfully rec_referenced_by_author {i}', flush=True)

    combined_recommendations = rec_cited_author(combined_recommendations, pulled_citations, authors_batch,
                                                references_df)
    combined_recommendations.to_csv(f"./Data/new_tfidf_batch_v3_7_19_2024/recommendations_authors_method_1000aut_max_{i}.csv")
    print('successfully rec_cited_author', flush=True)
    # print("authors_batch: ", authors_batch)
    print(f"Processed batch: {i}")

    return None


def create_batches(authors_list, batch_size):
    for i in range(0, len(authors_list), batch_size):
        yield i, authors_list[i:i + batch_size]  # Yield with index

batch_size = 100
authors_batches = list(create_batches(authors_list, batch_size))

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_authors_batch, idx, batch): idx for idx, batch in authors_batches}
    for future in concurrent.futures.as_completed(futures):
        batch_index = futures[future]
    print('All batches processed!')

