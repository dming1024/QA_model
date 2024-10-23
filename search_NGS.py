

# -*- coding: utf-8 -*-
"""
Created on 2024.10
@author: Qiangqiang Fan
Description: 主要用于检索PDX,细胞系,Syngeneic模型的突变、表达等数据
"""

# Import necessary modules
import os
import sys
import argparse  # For handling command-line arguments
import requests


def search(context):

    data_to_submit = {'input': context}
    response=requests.post('http://10.111.17.67:5000/predict', json=data_to_submit)

    # 检查响应状态码
    if response.status_code == 200:
        # 将文件保存到本地
        with open('search_result.tsv', 'wb') as f:
            f.write(response.content)
        print('文件已成功下载并保存为 search_result.tsv')
    else:
        print('请求失败，状态码:', response.status_code)


def main():
    parser = argparse.ArgumentParser(description="search NGS data based on your input")
    parser.add_argument('--question', type=str, required=True, help="your query question: 在我们PDX模型中检索EGFR突变数据")
    args = parser.parse_args()
    
    # Use the example function
    input_contest=args.question
    search(input_contest)

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()


'''
一些成功的示例：
python search_NGS.py --question 在pdx模型中找一下kras突变数据
python search_NGS.py --question 哪些细胞系模型中有egfr突变数据呢
python search_NGS.py --question 在小鼠模型中找一些kras突变数据
python search_NGS.py --question 在小鼠模型中找一些kras,egfr突变数据
'''