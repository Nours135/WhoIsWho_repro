import pandas as pd
import json
import re
import os

def format_author_name(author_name):
    # 移除中间名，只保留名和姓
    parts = author_name.split()
    if len(parts) > 2:
        parts = [parts[0], parts[-1]]  # 保留第一个和最后一个部分
    formatted_name = '_'.join(parts).lower()  # 转换为小写并用下划线连接
    return formatted_name

def create_author_name_map(excel_file):
    df = pd.read_excel(excel_file)
    author_name_map = {}
    for index, row in df.iterrows():
        author_id = row['author_id']
        original_name = row['author_name']
        formatted_name = format_author_name(original_name)
        author_name_map[author_id] = formatted_name
    return author_name_map

def replace_keys_in_json(input_json_file, output_json_file, name_map):
    with open(input_json_file, 'r') as f:
        data = json.load(f)
    
    new_data = {name_map[author_id]: value for author_id, value in data.items()}
    
    with open(output_json_file, 'w') as f:
        json.dump(new_data, f, indent=4)

def main():
    excel_file = 'original_excel.xlsx'
    input_json_file = 'train_author.json'
    output_json_file = 'modified_train_author.json'
    
    # 创建作者名字映射
    author_name_map = create_author_name_map(excel_file)
    
    # 替换JSON文件中的键
    replace_keys_in_json(input_json_file, output_json_file, author_name_map)
    
    print(f"Updated JSON file saved as {output_json_file}")

if __name__ == '__main__':
    main()