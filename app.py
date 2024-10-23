
#! /home_bk/fan_qiangqiang/miniconda3/envs/chatGLM/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2024.10.16
@author: fan_qiangqiang
Description: 在PDX，细胞系，Syngeneic模型中，检索基因的表达、突变、基因融合，拷贝数变异等数据
Update
"""

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# 初始化 Flask 应用
app = Flask(__name__)

# 在应用启动时加载模型
model = load_model('my_model.h5')  # 加载你的模型

# 创建推理函数
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求中的数据（假设为 JSON 格式）
        data = request.get_json()

        # 将数据转化为模型输入的格式，例如将其转为 NumPy 数组
        input_data = np.array(data['input'])

        # 使用模型进行预测
        predictions = model.predict(input_data)

        # 返回预测结果，转化为 Python 列表并以 JSON 格式返回
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
