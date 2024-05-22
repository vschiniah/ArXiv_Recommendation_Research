import pandas as pd

df = pd.read_csv('./Data/arxiv_metadata_dataset.csv')
categories = pd.read_csv('./Data/categories.csv')
final_rows = []
for ind, row in df.iterrows():
    row['Categories'] = row['categories'].split(' ')
    main_category = []
    sub_category = []
    for x in row['Categories']:
        if x in categories['ID'].values:
            main_category.append(categories.loc[categories['ID'] == x, 'Main Category'].values[0])
            sub_category.append(categories.loc[categories['ID'] == x, 'Name'].values[0])

    row['Main Category'] = ', '.join(list(set(main_category)))
    row['Sub Category'] = ', '.join(list(set(sub_category)))
    final_rows.append(row)

final_df = pd.DataFrame(final_rows)
cs_df = final_df[final_df['Main Category'].isin(['Computer Science'])]

cs_df['update_date'] = pd.to_datetime(cs_df['update_date'])
cs_df['year_updated'] = cs_df['update_date'].dt.year
test_data = cs_df[cs_df['year_updated'] == 2020]
test_data.to_csv('./Data/test_data.csv')

train_data = cs_df[cs_df['year_updated'] < 2020]
train_data.to_csv('./Data/train_data.csv')

