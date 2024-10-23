

'''
使用小批量数据，对模型进行预训练，主要用于应答数据库查询，根据客户问题提取关键字段：模型、基因、数据类型
'''


'''
数据预处理，将csv转换为squad2格式的数据
'''

from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from datasets import Dataset

model_name = "./models_roberta-base-chinese-extractive-qa/"  # 使用roberta-base-squad2模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

import csv
import json

def preprocess_function(examples):
    # 提取 question 和 context
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # 对 question 和 context 进行批量编码，设置最大长度和 padding
    inputs = tokenizer(
        questions,
        contexts,
        truncation="only_second",  # 在 context 进行截断
        max_length=512,  # 设置最大长度为 512
        padding="max_length",  # 使用最大长度进行填充
        return_offsets_mapping=True,  # 返回 offsets 以计算答案的位置
        return_tensors="np"  # 返回 NumPy 格式，确保所有长度一致
    )

    # 为了保存答案的起始和结束 token 索引，我们手动计算
    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"])):
        # 获取答案的开始字符位置
        answer_start = examples["answers"][i]["answer_start"]
        answer_text = examples["answers"][i]["text"]

        # 获取 context 的 offset 映射
        offset_mapping = inputs["offset_mapping"][i]
        input_ids = inputs["input_ids"][i]

        # 查找答案的起始和结束 token 索引
        start_char = answer_start
        end_char = start_char + len(answer_text)

        # 初始化 token 索引
        token_start_index = 0
        token_end_index = 0

        # 查找与字符索引对应的 token 索引
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                token_start_index = idx
            if start <= end_char <= end:
                token_end_index = idx
                break

        # 保存 token 索引
        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    # 移除 offset_mapping 因为我们已经不再需要它
    inputs.pop("offset_mapping")

    # 添加 start_positions 和 end_positions
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

def convert_to_squad_format(input_file, output_file):
    squad_data = {"data": []}
    
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        paragraphs = []
        for row in reader:
            context = row['context']
            question = row['question']
            answer_text = row['answers']
            
            # 找到答案在 context 中的起始位置
            answer_start = context.find(answer_text)
            
            if answer_start == -1:
                raise ValueError(f"Answer '{answer_text}' not found in context '{context}'")
            
            # 构建符合 SQuAD 格式的结构
            qas = {
                "question": question,
                "id": str(hash(question)),
                "answers": [{
                    "text": answer_text,
                    "answer_start": answer_start
                }],
                "is_impossible": False
            }
            
            paragraph = {
                "context": context,
                "qas": [qas]
            }
            
            paragraphs.append(paragraph)
        
        # 将段落添加到 "data" 部分
        squad_data["data"].append({
            "title": "custom_dataset",
            "paragraphs": paragraphs
        })
    
    # 将结果写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(squad_data, outfile, ensure_ascii=False, indent=4)

# 使用示例
convert_to_squad_format("QA_test_data/test.csv", "QA_test_data/test_squad_format.json")


# 加载 SQuAD 格式的 JSON 数据集
with open("QA_test_data/test_squad_format.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# 准备数据
contexts = []
questions = []
answers_text = []
answers_start = []

# 遍历数据集，将其转化为需要的格式
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            for answer in qa['answers']:
                contexts.append(context)
                questions.append(question)
                answers_text.append(answer['text'])
                answers_start.append(answer['answer_start'])

# 将数据加载为 Dataset 对象
data_dict = {
    'context': contexts,
    'question': questions,
    'answers': [{'text': a, 'answer_start': b} for a, b in zip(answers_text, answers_start)]
}
dataset = Dataset.from_dict(data_dict)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    evaluation_strategy="epoch",     # 每轮评估一次
    learning_rate=3e-4,              # 学习率
    per_device_train_batch_size=8,  # 每个设备的batch size
    per_device_eval_batch_size=8,   # 验证的batch size
    num_train_epochs=50,              # 训练轮数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./tmp/finetuned-roberta-base-squad2_wuxi1")
tokenizer.save_pretrained("./tmp/finetuned-roberta-base-squad2_wuxi1")

