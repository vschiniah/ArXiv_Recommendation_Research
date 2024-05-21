import pandas as pd
import pickle
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_authors, preprocess_text
from Scripts.model_eval_functions_v3 import rec_cited_author, rec_referenced_by_author

start_time = time.perf_counter()

# Import dataset
train_data = pd.read_csv('./Data/train_data.csv')

#  Test Data
test_data = pd.read_csv('./Data/test_data_s2.csv')

test_data['full_names'] = test_data['authors_parsed'].apply(extract_authors)
all_full_names = sum(test_data['full_names'], [])
unique_full_names = set(all_full_names)

stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read()
stop = stopwords.replace('\n', ',').split(",")

features = ['title', 'abstract', 'categories']

for feature in features:
    train_data[f'{feature}_processed'] = train_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))
    test_data[f'{feature}_processed'] = test_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))

features_processed = ['title_processed', 'abstract_processed', 'categories_processed']
# Combine features into a single string for further processing
train_data['combined_features'] = train_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                       axis=1)
test_data['combined_features'] = test_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                     axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_train_data = tfidf.fit_transform(train_data['combined_features'])
with open('tfidf_matrix_train_data.pickle', 'wb') as fin:
    pickle.dump(tfidf_matrix_train_data, fin)

tfidf_matrix_test_data = tfidf.transform(test_data['combined_features'])
with open('tfidf_matrix_test_data.pickle', 'wb') as fin:
    pickle.dump(tfidf_matrix_test_data, fin)

print('successfully created embeddings for train and test data')


def recommend_papers(author, train_data, test_data=None, tfidf_matrix_train_data=None, tfidf_matrix_test_data=None):
    """
    Recommends papers for given author
    :param author: list of authors
    :param train_data: dataset of papers before 2020
    :param test_data: dataset of papers in 2020
    :param tfidf_matrix_train_data: tfidf matrix of train data
    :param tfidf_matrix_test_data: tfidf matrix of test data
    :return:
    """
    author_df = train_data[train_data['authors'].apply(lambda authors: author in authors)]
    trained_on = (author, len(author_df))

    if not author_df.empty:
        # Creating the TF-IDF Vectorizer and computing the cosine similarity matrix
        author_indices = author_df.index.tolist()
        authors_matrix = tfidf_matrix_train_data[author_indices]
        with open(f'{author}_matrix.pickle', 'wb') as fin:
            pickle.dump(authors_matrix, fin)
        print(f'successfully pulled embedding for {author} from train data')
        cosine_sim = cosine_similarity(authors_matrix, tfidf_matrix_test_data)
        avg_cosine_sim = cosine_sim.max(axis=0)
        scores_per_paper = {test_data.iloc[i]['title']: score for i, score in enumerate(avg_cosine_sim)}

        avg_sim_scores = []

        # Iterate over the author's papers in the cosine_sim matrix
        for jdx, score in enumerate(avg_cosine_sim):
                avg_sim_scores.append((jdx, score))

        # Sort papers based on similarity scores
        avg_sim_scores = sorted(avg_sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 10 recommendations (excluding author's own papers)
        # top_paper_indices = [i[0] for i in avg_sim_scores[:10]]
        all_paper_indices = [i[0] for i in avg_sim_scores]

        recommended_df = test_data.iloc[all_paper_indices][['id', 'title', 'authors', 'abstract', 'authors_parsed',
                                                            's2PaperId', 'corpusId', 'year']]
        recommended_df['recommended_to'] = author
        recommended_df['number of papers trained on'] = trained_on[1]
        return recommended_df, scores_per_paper
    else:
        return pd.DataFrame(columns=['id', 'title', 'authors', 'abstract', 'authors_parsed']), {}


def recommend_for_authors(authors_list, train_data, test_data=None, tfidf_matrix_train_data=None,
                          tfidf_matrix_test_data=None):
    """
    Function that calls recommendations for given author
    :param authors_list:
    :param train_data:
    :param test_data:
    :param tfidf_matrix_train_data:
    :param tfidf_matrix_test_data:
    :return:
    """
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

        if all_recommendations.empty:
            all_recommendations['similarity_score'] = []
        else:
            matched_scores = []
            for index, row in all_recommendations.iterrows():
                title = row['title']
                if title in scores_per_paper:
                    matched_scores.append(scores_per_paper[title])
                else:
                    matched_scores.append(float(0))
            all_recommendations['similarity_score'] = matched_scores
        print(f'successfully recommended for {author}')
    scores_df = pd.DataFrame(scores_data)
    scores_df = scores_df.set_index('author')

    return all_recommendations, scores_df


references_df = pd.read_csv('../papers_and_authors.csv')
pulled_references = pd.read_csv('../pulled_references.csv')
pulled_citations = pd.read_csv('../pulled_citations.csv')
num_authors_to_select = min(1000, len(unique_full_names))
authors_list = random.sample(sorted(unique_full_names), num_authors_to_select)

all_recommendations, scores_df = recommend_for_authors(authors_list, train_data, test_data, tfidf_matrix_train_data,
                                                       tfidf_matrix_test_data)
print('successfully created recommendations')
combined_recommendations = rec_referenced_by_author(all_recommendations, pulled_references, authors_list, references_df)
print('successfully rec_referenced_by_author and co-authored')
combined_recommendations = rec_cited_author(combined_recommendations, pulled_citations, authors_list, references_df)
print('successfully rec_cited_author')

scores_df.to_csv("average_author_similarity_scores.csv")
combined_recommendations.to_csv("recommendations_authors_method.csv")

print(f"Finished all the recommendations for {num_authors_to_select} authors")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
