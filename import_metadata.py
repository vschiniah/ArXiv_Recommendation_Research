import pandas as pd
import json
import os
from tqdm.notebook import tqdm


def get_metadata():
    with open('research/arxiv-metadata-oai-snapshot.json') as f:
        for line in f:
            yield line


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

metadata = get_metadata()

for paper in metadata:
    first_paper = json.loads(paper)
    break

for key in first_paper:
    print(key)

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

metadata = get_metadata()
total_items = 0

for ind, paper in enumerate(metadata):
    paper = json.loads(paper)
    total_items += 1

    ids.append(paper['id'])
    submitter.append(paper['submitter'])
    authors.append(paper['authors'])
    title.append(paper['title'])
    comments.append(paper['comments'])
    journal_ref.append(paper['journal-ref'])
    doi.append(paper['doi'])
    report_no.append(paper['report-no'])
    categories.append(paper['categories'])
    license.append(paper['license'])
    abstract.append(paper['abstract'])
    versions.append(paper['versions'])
    update_date.append(paper['update_date'])
    authors_parsed.append(paper['authors_parsed'])

print(f'Total number of items is: {total_items}')

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

df.to_csv('arxiv_metadata_dataset.csv')
