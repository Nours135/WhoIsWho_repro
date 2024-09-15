# convert author data to bond format
import json
import multiprocessing
import os
import pickle
import re
import time
from typing import Dict
import pandas as pd
import hashlib
from functools import wraps
import requests as rq
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class CacheDecorator:
    def __init__(self, cache_dir='./cached'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique cache file name based on function name and arguments
            
            # TODO: further improvement
            # 根据函数名和hash值范围分箱存储
            # print(str(args).encode() + str(kwargs).encode())
            # raise Exception('debug')
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}.pickle"
            cache_file = os.path.join(self.cache_dir, cache_key)
            
            # Check if cache file exists
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    # print(f'Loading result from cache file: {cache_file}')
                    return pickle.load(f)
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Write result to cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
                # print(f'Saving result to cache file: {cache_file}')
            
            return result
        return wrapper
    
    def rm_cache(self):
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))
        print(f'All cache files removed from {self.cache_dir}')
    

@CacheDecorator()
@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=4, max=30),  # 指数退避策略，初始等待1秒，最大等待10秒
    retry=retry_if_exception_type(rq.exceptions.RequestException)  # 仅在请求异常时重试
)
def get_work_info_from_openalex(workid: str) -> Dict[str, any]:
    '''
    Query openalex for extra work info
    '''
    url = f'https://api.openalex.org/works/{workid}'
    response = rq.get(url)
    if response.status_code == 200:
        res = response.json()
        assert workid == res['id'].split('/')[-1], f'mismatch, workid: {workid}, res_id: {res["id"]}'
        
        # process abs
        try:
            abstract_index_2_text = dict()
            for token, inverted_idx_l in res['abstract_inverted_index'].items():
                for idx in inverted_idx_l:
                    abstract_index_2_text[idx] = token
            output_abs = []
            for idx in range(min(abstract_index_2_text.keys()), max(abstract_index_2_text.keys())+1):
                if idx not in abstract_index_2_text:
                    print(f'missing idx: {idx} in abstract_inverted_index')
                output_abs.append(abstract_index_2_text.get(idx, ''))
            output_abs = ' '.join(output_abs)
        except AttributeError:
            print(f'fail to get abstract of {workid}')
            print(res.keys())
            output_abs = ''
        except ValueError:  # ValueError: min() arg is an empty sequence
            output_abs = ''
        # print(output_abs)
        
        # process authorships
        author_info_l = []
        for authorship in res['authorships']:

            if len(authorship['institutions']) > 0:
                org = authorship['institutions'][0]['display_name']
            else:
                org = ''
            author_info_l.append({
                'name': authorship['author']['display_name'],
                'org': org
            })
        # print(res['authorships'][0])
        # print(res.keys())
        try:
            venue = res['primary_location']['source']['display_name']
        except TypeError:
            venue = ''
        output_key_info = {
            'id': workid,
            'title': res['title'],
            'abstract': output_abs,
            'keywords': [item['display_name'] for item in res['keywords'] + res['concepts'] if item['score'] > 0.5],
            'authors': author_info_l,  
            'venue': venue,
            'year': res['publication_year'],
        }
    else:
        raise Exception(f'fail to get work info of {workid}')
    time.sleep(10)
    return output_key_info

def is_valid_workid(wid: str) -> bool:
    return bool(re.match(r'^W\d+$', wid))

def generate_author_data():
    
    pub_set = set()
    
    df = pd.read_excel('./groundtruth_0813.xlsx')
    df['author_id'] = df['author_id_1'].apply(lambda x: x.split('_')[0])
    print(f'coun before dropna: {len(df)}')
    df = df.dropna(subset=['work_id'])
    # print(df.head(5))
    print(f'count after dropna: {len(df)}')
    
    author_gt_data = dict()
    for author_id, author_df in df.groupby('author_id'):
        author_id_1_to_pub_l = dict()
        for aid_1, aid1_df in author_df.groupby('author_id_1'):
            work_l = [item.split('/')[-1].replace('w', 'W') for item in aid1_df['work_id'].tolist()]
            work_l = [item for item in work_l if is_valid_workid(item)]
            author_id_1_to_pub_l[aid_1] = work_l
            pub_set = pub_set | set(work_l)
        author_gt_data[author_id] = author_id_1_to_pub_l
        
    with open('./src/total/train_author.json', 'w') as f:
        json.dump(author_gt_data, f, indent=4)
    print(f'dumped ./src/total/train_author.json')
    return pub_set

def generate_work_data(pub_set):
    '''
    abs, keywords, venue, author and raw_institution_str needed to be extracted
    
    '''
    pool = multiprocessing.Pool(5)
    total_pubs = dict()
    for work_info in tqdm(pool.imap_unordered(get_work_info_from_openalex, list(pub_set)), desc='get work info', total=len(pub_set)):
        total_pubs[work_info['id']] = work_info
    
    with open('./src/total/train_pub.json', 'w') as f:
        json.dump(total_pubs, indent=4, fp=f)
    print(f'dumped ./src/total/train_pub.json')

def main():
    if not os.path.exists('./src'): os.makedirs('./src')
    if not os.path.exists('./src/total'): os.makedirs('./src/total')
    
    # step 1 convert and generate train_auhtors.json
    pub_set = generate_author_data()
    
    # step 2
    generate_work_data(pub_set)
    
if __name__ == '__main__':
    main()
    # res = get_work_info_from_openalex('W3017091631')
    # print(json.dumps(res, indent=4))
    # cacher = CacheDecorator()
    # cacher.rm_cache()