import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, AutoModel
from annoy import AnnoyIndex
from tqdm import tqdm
import os

# 设置环境
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保GPU正确排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("ernie-3.0-base-zh")
model = AutoModel.from_pretrained("ernie-3.0-base-zh").to(device)

def encode_texts(texts, batch_size=1000):
    """文本向量化"""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
        batch_texts = texts[i:i + batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())
    return np.vstack(all_embeddings)

def build_annoy_index(embeddings, n_trees=100):
    """使用Annoy构建索引"""
    dim = embeddings.shape[1]
    index = AnnoyIndex(dim, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(n_trees)
    return index

def detect_templated_expressions(index, embeddings, top_k=6):
    """检测模板化表达"""
    n_items = embeddings.shape[0]
    max_similarities = []
    for i in tqdm(range(n_items), desc="Detecting Templated Expressions"):
        neighbors = index.get_nns_by_item(i, top_k, include_distances=True)
        similarities = [1 - (distance / 2.0) for distance in neighbors[1][1:]]  # 距离转相似度
        max_similarity = max(similarities)
        max_similarities.append(max_similarity)
    return max_similarities

def compute_comment_length(text):
    """计算评论长度"""
    return len(text)

# 读取Excel文件
file_path = '分析结果.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 添加评论长度列
data['评论长度'] = data['发言文本'].apply(compute_comment_length)

# 向量化评论文本
texts = data['发言文本'].tolist()
embeddings = encode_texts(texts)

# 构建Annoy索引并检测相似度
annoy_index = build_annoy_index(embeddings)
data['最大相似度'] = detect_templated_expressions(annoy_index, embeddings)

# 基于评分与评论内容的匹配度进行筛选
def match_score_sentiment(row):
    if row['情感倾向'] == '积极' and row['评分'] <= 2:
        return True
    if row['情感倾向'] == '消极' and row['评分'] >= 4:
        return True
    return False

data['匹配度异常'] = data.apply(match_score_sentiment, axis=1)

# 标记相似评论
data['相似评论'] = data['最大相似度'] > 0.99

# 标记虚假评论
data['虚假评论'] = ((data['评论长度'] < 3) & (data['情感倾向'] == '积极')) | \
                   (data['匹配度异常']) | \
                   (data['相似评论'])

# 创建虚假评论标记列
data['虚假评论标记'] = data['虚假评论'].apply(lambda x: '可能虚假' if x else '可能真实')

# 保存结果到新的Excel文件
output_file_path = '虚假评论结果.xlsx'  # 替换为你希望保存的文件路径
data.to_excel(output_file_path, index=False)

print(f"虚假评论结果已保存到 {output_file_path}")
