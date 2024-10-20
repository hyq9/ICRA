# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, user_log_path, course_text_path, pretrained_model_path, max_len=256):
        self.course_text_df = pd.read_excel(course_text_path)
        self.logs_user_df = pd.read_excel(user_log_path).head(100)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.logs_user_df)

    def __getitem__(self, idx):
        user_log = self.logs_user_df.iloc[idx]
        course_text = self.course_text_df[self.course_text_df['课程ID'] == user_log['课程ID']]['课程概述'].values[0]
        text = user_log['发言文本'] + " [SEP] " + course_text
        inputs = self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(user_log['评分'], dtype=torch.float)
        }

def create_dataloaders(dataset, batch_size=8):
    train, val = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 模型定义
class RatingPredictorModel(nn.Module):
    def __init__(self, pretrained_model_path):
        super(RatingPredictorModel, self).__init__()
        self.ernie = AutoModel.from_pretrained(pretrained_model_path)
        self.self_attention = nn.MultiheadAttention(embed_dim=self.ernie.config.hidden_size, num_heads=2)
        self.attention_norm = nn.LayerNorm(self.ernie.config.hidden_size)
        self.attention_dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(self.ernie.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        ernie_outputs = self.ernie(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = ernie_outputs.last_hidden_state[:, 0, :].unsqueeze(1)
        attention_output, _ = self.self_attention(cls_output, cls_output, cls_output)
        attention_output = self.attention_norm(cls_output + self.attention_dropout(attention_output))
        mlp_output = self.mlp(attention_output.squeeze(1))
        return mlp_output

# 训练模型
def train_model(model, train_loader, val_loader, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_losses, val_rmses, val_maes = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = nn.MSELoss()(outputs.squeeze(), batch['labels'].to(device))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_mse_sum, val_loss_mae_sum = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
                val_loss_mse_sum += nn.MSELoss()(outputs, labels).item()
                val_loss_mae_sum += nn.L1Loss()(outputs, labels).item()

        avg_val_loss_mse = val_loss_mse_sum / len(val_loader)
        avg_val_loss_mae = val_loss_mae_sum / len(val_loader)
        valid_rmse = avg_val_loss_mse ** 0.5
        val_rmses.append(valid_rmse)
        val_maes.append(avg_val_loss_mae)

        print(f"Epoch {epoch + 1:3d}; train loss {avg_train_loss:.6f}; validation rmse {valid_rmse:.6f}, validation mae {avg_val_loss_mae:.6f}")
    # 计算MAE和RMSE的总体平均值
    avg_val_rmse = sum(val_rmses) / len(val_rmses)
    avg_val_mae = sum(val_maes) / len(val_maes)
    print(f"\nOverall validation RMSE: {avg_val_rmse:.6f}, Overall validation MAE: {avg_val_mae:.6f}")



if __name__ == "__main__":
    pretrained_model_path = 'ernie-3.0-base-zh'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(user_log_path='./data/用户.xlsx', course_text_path='./data/课程.xlsx', pretrained_model_path=pretrained_model_path)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=8)

    model = RatingPredictorModel(pretrained_model_path=pretrained_model_path).to(device)
    train_model(model, train_loader, val_loader, epochs=50, learning_rate=5e-5)
