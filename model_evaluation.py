from utils import convert_name


def rec_referenced_by_author(all_recommendations, pulled_references, authors_list, stored_references):
    """
    This function checks whether the recommended papers reference the author's work
    and if the author wrote the recommended paper
    :param all_recommendations: recommended papers
    :param pulled_references: references pulled from semantic scholar for all the papers
    :param authors_list: authors list that papers were recommended to
    :param stored_references: all authors and papers each author wrote
    :return: dataframe with recommendations and binary classification for references and co-authorship
    """

    all_recommendations["References Author"] = False
    all_recommendations["Also Authored by input author"] = False

    refs_dict = {
        source_id: group.drop('Source Paper ID', axis=1).to_dict('records')
        for source_id, group in pulled_references.groupby('Source Paper ID')
    }

    for author_initials in authors_list:
        converted_authors_list = convert_name(author_initials)
        recs = all_recommendations[all_recommendations['recommended_to'] == author_initials]
        filtered_papers_df = stored_references[
            ((stored_references['Author Name'] == author_initials) |
             (stored_references['Author Name'] == converted_authors_list))
        ]

        for idx, rec_row in recs.iterrows():
            # Filter for references where the 'Source Paper Title' matches the recommendation title
            filt_data = refs_dict[rec_row['s2PaperId']]
            filt_paper_ids = [ref['Reference Paper ID'] for ref in filt_data]

            is_referenced = filtered_papers_df['Paper ID'].isin(filt_paper_ids)

            if is_referenced.any():
                recs.loc[idx, 'References Author'] = True

            is_present = any(filtered_papers_df['Paper ID'].str.contains(rec_row['s2PaperId'], case=False, na=False))

            if is_present:
                recs.loc[idx, "Also Authored by input author"] = True

        all_recommendations.loc[recs.index, :] = recs

    return all_recommendations


def rec_cited_author(all_recommendations, pulled_citations, authors_list, stored_references):
    """
    Checks if any of the recommendations were later cited by the authors. This function cross-references
    pulled citations with author-specific stored references to identify matches.

    :param all_recommendations: DataFrame containing all recommended papers for all the different authors.
    :param pulled_citations: DataFrame containing citations for all recommended papers.
    :param authors_list: List of authors for recommended papers.
    :param stored_references: DataFrame containing papers authored by the authors.
    :return: Updated DataFrame with citation information.
    """

    all_recommendations['Citations Present'] = 'No Citations'
    for author_initials in authors_list:
        converted_authors_list = convert_name(author_initials)

        author_papers = stored_references[
            (stored_references['Author Name'].isin([author_initials, converted_authors_list]))
        ]
        authored_paper_ids = author_papers['Paper ID'].unique()

        citations_by_author = pulled_citations[pulled_citations['Citation Paper ID'].isin(authored_paper_ids)]

        cited_recommendations_ids = citations_by_author['Source Paper ID'].unique()

        for rec_id in cited_recommendations_ids:
            all_recommendations.loc[
                (all_recommendations['s2PaperId'] == rec_id) & (
                            all_recommendations['recommended_to'] == author_initials),
                'Citations Present'
            ] = True

    return all_recommendations
