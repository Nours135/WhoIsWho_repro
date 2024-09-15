# descriptive stats of this dataset

import os
import pandas as pd
from pic_utils import draw_hist
# stats 1, author and pub count
def author_pub_count():
    '''
    count of gt authors and count
    '''
    df = pd.read_excel('./groundtruth_0813.xlsx')
    df['author_id'] = df['author_id_1'].apply(lambda x: x.split('_')[0])
    author_id_1_l = df['author_id_1'].tolist()
    # get the origin author_id 
    author_id_l = df['author_id'].tolist()
    
    pub_count = len(set(df['work_id'].tolist()))
    print(f'Origin author count: {len(set(author_id_l))}')
    print(f'Disambiguated author count: {len(set(author_id_1_l))}')
    print(f'Publication count: {pub_count}')
    # return author_count, pub_count

def pub_distribution():
    df = pd.read_excel('./groundtruth_0813.xlsx')
    df['author_id'] = df['author_id_1'].apply(lambda x: x.split('_')[0])
    
    if not os.path.exists('./figs'):
        os.makedirs('./figs')
    author_pub_l = []
    for author, group_df in df.groupby('author_id'):
        author_pub_l.append(len(group_df))
    draw_hist(author_pub_l, title='Publication per author_id', xlabel='Publication count', ylabel='Author count', save_path='./figs/pub_per_author_id.png')
    
    author_id_1_pub_l = []
    for author, group_df in df.groupby('author_id_1'):
        pub_c = len(group_df)
        author_id_1_pub_l.append(pub_c)
        if pub_c > 500:
            print(f'author_id_1: {author}, pub count: {pub_c}')
    draw_hist(author_id_1_pub_l, title='Publication per author_id_1', xlabel='Publication count', ylabel='Author count', save_path='./figs/pub_per_author_id_1.png')
    
    
if __name__ == '__main__':
    # stats 1, author and pub count
    author_pub_count()
    
    # stats 2, pub per author distribution
    pub_distribution()