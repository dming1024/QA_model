#! /home_bk/fan_qiangqiang/miniconda3/envs/chatGLM/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2024.10.16
@author: fan_qiangqiang
Description: 在PDX，细胞系，Syngeneic模型中，检索基因的表达、突变、基因融合，拷贝数变异等数据
Update: 更新为API使用
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering,pipeline
import subprocess
import argparse
from flask import Flask, request, jsonify, send_file


def get_dict(filein):
    '获取model_types 和 datatypes'
    dict_map={}
    with open(filein,'r') as f:
        fs=f.readlines()
        for rs in fs:
            records=rs.split("\t")
            dict_map[records[0]]=records[1].strip("\n")
    return dict_map


def search(models,genes,datatypes):
    '根据输入的models，genes和datatypes，在服务器后台进行检索'
    dt=datatype_dict[datatypes]
    genes="|".join([m.upper() for m in genes.split(",")])

    #还需要先评价下是否有空值

    if dt=='mutation':
        search_mutation(models,genes,dt)

    if dt=='expression':
        search_expression(models,genes,dt)

    if dt=='CNV':
        search_CNV(models,genes,dt)

    if dt=='fusion':
        search_fusion(models,genes,dt)

    if dt=='HLA':
        search_HLA(models,genes,dt)


def search_mutation(models,genes,dt):
    '检索突变'
    command="cat {NGSpath}*|grep -E exonic|grep -wiE '{genes}'".format(
        NGSpath=getattr(tmp_database[model_dict[models]],dt),
        genes= genes)
    #print(command)
    result = subprocess.Popen(command,shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    stdout, stderr = result.communicate()
    with open('res.tsv','w') as f:
        f.writelines("\t".join(["Model ID","WGC ID","chroM","Start","End","ref","Alt",
                               "Genotype", "GeneLocation","Gene","Exon Syno",
                               "Exon","dbSNP", "1000G","Cosmic","AF","DP"])+"\n")
        f.writelines(stdout)

def search_expression(models,genes,dt):
    '检索表达'
    command="cat {NGSpath}*|grep -wiE '{genes}'|cut -f1,2,5".format(
        NGSpath=getattr(tmp_database[model_dict[models]],dt),
        genes= genes)
    #print(command)
    result = subprocess.Popen(command,shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    stdout, stderr = result.communicate()
    with open('res.tsv','w') as f:
        f.writelines("\t".join(['modelID','Gene','expression'])+"\n")
        f.writelines(stdout)

def search_CNV(models,genes,dt):
    '检索CNV'
    command="cat {NGSpath}*|grep -wiE '{genes}'|cut -f1,2,5".format(
        NGSpath=getattr(tmp_database[model_dict[models]],dt),
        genes= genes)
    #print(command)
    result = subprocess.Popen(command,shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    stdout, stderr = result.communicate()
    with open('res.tsv','w') as f:
        f.writelines("\t".join(["Model ID","Gene","CNV"])+"\n")
        f.writelines(stdout)

def search_fusion(models,genes,dt):
    '检索融合'
    command="cat {NGSpath}*|grep -wiE '{genes}' | cut -f1,2,7,8,9,10,14".format(
        NGSpath=getattr(tmp_database[model_dict[models]],dt),
        genes= genes)
    #print(command)
    result = subprocess.Popen(command,shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    stdout, stderr = result.communicate()
    with open('res.tsv','w') as f:
        f.writelines("\t".join(["ID","JunctionReadCount","LeftGene",
                                "LeftBreakpoint","RightGene","RightBreakpoint","FFPM"])+"\n")
        f.writelines(stdout)

def search_HLA(models,genes,dt):
    pass

def get_inputs(context_query):
    '从输入中获取模型，基因，数据类型'
    QA_input={'question':'什么模型','context':context_query}
    global model
    res = model(QA_input)
    models=res['answer'] if res['score'] >0.9 else ''

    QA_input={'question':'什么数据类型','context':context_query}
    res = model(QA_input)
    datatypes=res['answer'] if res['score'] >0.9 else ''

    QA_input={'question':'什么基因','context':context_query}
    res = model(QA_input)
    genes=res['answer'] if res['score'] >0.9 else ''

    return models,genes,datatypes

def write_out(log):
    '输出检索日志'
    with open('QA_log','a') as f:
        f.writelines(log+"\n")

class innerDatabse:
    def __init__(self,name,expression,mutation,CNV,HLA,fusion):
        self.name=name
        self.expression=expression
        self.mutation=mutation
        self.CNV=CNV
        self.HLA=HLA
        self.fusion=fusion

tmp_database={}
tmp_database['PDX']=innerDatabse(name='PDX',
                                 expression='/OIU/innerNGSresults/filterRNAseq/',
                                mutation='/OIU/innerNGSresults/filterWXS/',
                                CNV='/OIU/innerNGSresults/CNV_cnvkit/',
                                HLA='/OIU/innerNGSresults/HLA/',
                                fusion='/OIU/innerNGSresults/fusion_starfusion_formatPDX/')
tmp_database['cancer_cell_line']=innerDatabse(name='cancer cell line',
                                 expression='/OIU/innerNGSresults/CellLineDatasets/expression_format_TPM/',
                                mutation='/OIU/innerNGSresults/CellLineDatasets/mutation_format/',
                                CNV='/OIU/innerNGSresults/CellLineDatasets/cnv_format/',
                                HLA='/OIU/innerNGSresults/CellLineDatasets/HLA_HD/',
                                fusion='/OIU/innerNGSresults/CellLineDatasets/fusion_starfusion/')

tmp_database['syngeneic']=innerDatabse(name='syngeneic models',
                                 expression='/OIU/innerNGSresults/Syngeneic/expression/',
                                mutation='/OIU/innerNGSresults/Syngeneic/mutation/',
                                CNV='',
                                HLA='',
                                fusion='')

model = pipeline("question-answering",
                 model="./tmp/finetuned-roberta-base-squad2_wuxi1",
                 tokenizer="./tmp/finetuned-roberta-base-squad2_wuxi1")
model_dict=get_dict('./model_dict')
datatype_dict=get_dict('./datatype_dict')

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def predict():
    # Command-line argument parser
    #parser = argparse.ArgumentParser(description="search NGS data based on your input")
    #parser.add_argument('--question', type=str, required=True, help="your query question: 在我们PDX模型中检索EGFR突变数据")
    #args = parser.parse_args()

    # Use the example function
    #input_contest=args.question
    # 获取请求中的数据（假设为 JSON 格式）
    data = request.json
    #print(data)
    print(data['input'])#input_contest=data['input']
    input_contest=data['input']
    #return 'Data received', 200

    assert input_contest, "Your input not avaliable, please check your question"
    models,genes,datatypes=get_inputs(input_contest)

    res = 1 if models in model_dict and genes and datatypes in datatype_dict else 0
    if res:
        search(models,genes,datatypes)
        check_status="\t".join([input_contest,datatype_dict[datatypes],models,genes,datatypes,str(res)])
        write_out(check_status)
    else:
        check_status="\t".join([input_contest,'-',models,genes,datatypes,str(res)])
        write_out(check_status)
        
    return send_file('/home/fan_qiangqiang/home_bk/QA/res.tsv',as_attachment=True)

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    #main()
    app.run(host='0.0.0.0', port=5000)


'''
客户端代码实现：
import requests
 
data_to_submit = {
    'input': '请在小鼠模型检索下ABC的突变数据'
}
# 发送 POST 请求并获取响应
response=requests.post('http://localhost:5000/predict', json=data_to_submit)

# 检查响应状态码
if response.status_code == 200:
    # 将文件保存到本地
    with open('search_result.tsv', 'wb') as f:
        f.write(response.content)
    print('文件已成功下载并保存为 search_result.tsv')
else:
    print('请求失败，状态码:', response.status_code)
'''