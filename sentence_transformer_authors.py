import pandas as pd
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_authors
from model_evaluation import rec_referenced_by_author, rec_cited_author


# Import dataset
train_data = pd.read_csv('./Data/train_data.csv')


#  Test Data
test_data = pd.read_csv('./Data/test_data.csv')

test_data['full_names'] = test_data['authors_parsed'].apply(extract_authors)
all_full_names = sum(test_data['full_names'], [])
unique_full_names = set(all_full_names)

with open('./Data/train_matrix_transformer.pickle', 'rb') as f:
    tfidf_matrix_train_data = pickle.load(f)

with open('./Data/test_matrix_transformer.pickle', 'rb') as r:
    tfidf_matrix_test_data = pickle.load(r)


def recommend_papers(author, train_data, test_data=None, tfidf_matrix_train_data=None, tfidf_matrix_test_data=None):

    author_df = train_data[train_data['authors'].apply(lambda authors: author in authors)]
    trained_on = (author, len(author_df))

    if not author_df.empty:
        author_indices = author_df.index.tolist()
        authors_matrix = tfidf_matrix_train_data[author_indices]
        with open(f'./Data/{author}_matrix.pickle', 'wb') as fin:
            pickle.dump(authors_matrix, fin)
        print(f'successfully pulled embedding for {author} from train data')
        cosine_sim = cosine_similarity(authors_matrix, tfidf_matrix_test_data)
        avg_cosine_sim = cosine_sim.max(axis=0)
        scores_per_paper = {test_data.iloc[i]['title']: score for i, score in enumerate(avg_cosine_sim)}

        avg_sim_scores = []

        # Iterate over the author's papers in the cosine_sim matrix
        for jdx, score in enumerate(avg_cosine_sim):
            avg_sim_scores.append((jdx, score))

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
num_authors_to_select = min(1000, len(unique_full_names))
authors_list = random.sample(sorted(unique_full_names), num_authors_to_select)


all_recommendations, scores_df = recommend_for_authors(authors_list, train_data, test_data,
                                                       tfidf_matrix_train_data,
                                                       tfidf_matrix_test_data)
all_recommendations.to_csv(f"./Data/all_recommendations_authors_method_1000aut_max.csv")
scores_df.to_csv(f"./Data/average_author_similarity_scores_1000aut_max.csv")
print(f'successfully created recommendations', flush=True)

combined_recommendations = rec_referenced_by_author(all_recommendations, pulled_references, authors_list,
                                                    references_df)
print(f'successfully rec_referenced_by_author', flush=True)

combined_recommendations = rec_cited_author(combined_recommendations, pulled_citations, authors_list,
                                            references_df)
combined_recommendations.to_csv(f"./Data/recommendations_authors_method_1000aut_max.csv")
print('successfully rec_cited_author', flush=True)