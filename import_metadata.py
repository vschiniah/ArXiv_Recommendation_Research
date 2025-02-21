import pandas as pd
import json
import os
from tqdm import tqdm


def get_metadata():
    with open('Data/arxiv-metadata-oai-snapshot.json') as f:
        for line in f:
            yield line


metadata = get_metadata()

# Initialize the progress bar with the exact total number of items
total_items = 2666751 # length of metadata known!
progress_bar = tqdm(metadata, total=total_items)

ids = []
submitter = []
authors = []
title = []
comments = []
journal_ref = []
doi = []
report_no = []
categories = []
license = []
abstract = []
versions = []
update_date = []
authors_parsed = []

for paper in progress_bar:
    paper = json.loads(paper)

    ids.append(paper['id'])
    submitter.append(paper['submitter'])
    authors.append(paper['authors'])
    title.append(paper['title'])
    comments.append(paper['comments'] if 'comments' in paper else None)
    journal_ref.append(paper['journal-ref'] if 'journal-ref' in paper else None)
    doi.append(paper['doi'] if 'doi' in paper else None)
    report_no.append(paper['report-no'] if 'report-no' in paper else None)
    categories.append(paper['categories'])
    license.append(paper['license'] if 'license' in paper else None)
    abstract.append(paper['abstract'])
    versions.append(paper['versions'])
    update_date.append(paper['update_date'])
    authors_parsed.append(paper['authors_parsed'])

    # Update the progress description to show the current paper's ID
    progress_bar.set_description(f"Processing ID: {paper['id']}")

print(f'Total number of items processed: {len(ids)}')

d = {
    'id': ids,
    'submitter': submitter,
    'authors': authors,
    'title': title,
    'comments': comments,
    'journal-ref': journal_ref,
    'doi': doi,
    'report-no': report_no,
    'categories': categories,
    'license': license,
    'abstract': abstract,
    'versions': versions,
    'update_date': update_date,
    'authors_parsed': authors_parsed
}

df = pd.DataFrame(d)

df.to_csv('./Data/arxiv_metadata_dataset.csv', index=False)