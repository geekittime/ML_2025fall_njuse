import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 导入 PyTorch 相关库 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 设置随机种子以保证结果可复现 ---
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 忽略一些未来版本的警告
warnings.filterwarnings('ignore')

# --- 1. 数据加载与合并 (保持不变) ---
print("步骤 1: 开始加载和合并数据...")
# project_name = input("使用的项目目录 (例如 yii2): ")
project_name = "yii2"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
path_prefix = os.path.join(parent_dir, project_name)

print(f"正在从以下路径加载数据: {path_prefix}")
extracted_path = os.path.join(path_prefix, 'PR_extracted_features.xlsx')
merged_df = pd.read_excel(extracted_path)

print(f"数据加载完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

# --- 2. 数据预处理与特征工程 (保持不变) ---
print("\n步骤 2: 开始数据预处理和特征工程...")

# (数据处理部分与之前完全相同)
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
merged_df['closed_at'] = pd.to_datetime(merged_df['closed_at'])
merged_df['TTC_hours'] = (merged_df['closed_at'] - merged_df['created_at']).dt.total_seconds() / 3600
merged_df.dropna(subset=['closed_at', 'created_at'], inplace=True)
merged_df = merged_df[merged_df['TTC_hours'] >= 0]
merged_df['log_TTC_hours'] = np.log1p(merged_df['TTC_hours'])
for bool_col in merged_df.select_dtypes(include=bool).columns.tolist():
    merged_df[bool_col] = merged_df[bool_col].astype(int)
features = merged_df.select_dtypes(include=np.number).columns.tolist()
features_to_remove = ['number', 'TTC_hours', 'log_TTC_hours']
features = [f for f in features if f not in features_to_remove]
X = merged_df[features]
y = merged_df['log_TTC_hours']
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print(f"最终选定的特征数量: {len(features)}")

# --- 3. 按时间顺序划分与标准化 (保持不变) ---
print("\n步骤 3: 按时间顺序划分与标准化...")
merged_df.sort_values('created_at', inplace=True)
X = X.loc[merged_df.index]
y = y.loc[merged_df.index]

split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集大小: {X_train.shape[0]} 行")
print(f"测试集大小: {X_test.shape[0]} 行")

# --- 4. 准备 PyTorch 数据 (保持不变) ---
print("\n步骤 4: 准备 PyTorch 数据...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用 {device} 设备进行训练。")
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).to(device).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).to(device).view(-1, 1)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 5. 修改点：定义 Wide&Deep 神经网络模型 ---
print("\n步骤 5: 定义 Wide&Deep 神经网络模型...")


class WideAndDeepModel(nn.Module):
    def __init__(self, input_dim):
        super(WideAndDeepModel, self).__init__()

        # Wide部分：一个简单的线性层，用于“记忆”
        self.wide_part = nn.Linear(input_dim, 1)

        # Deep部分：一个MLP，用于“泛化”
        self.deep_part = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Deep部分的输出也是1维
        )

    def forward(self, x):
        # 分别计算 wide 和 deep 部分的输出
        wide_output = self.wide_part(x)
        deep_output = self.deep_part(x)

        # 将两部分的结果相加得到最终输出
        return wide_output + deep_output


# 实例化模型
input_dim = X_train_scaled.shape[1]
model = WideAndDeepModel(input_dim).to(device)
print(model)

# --- 6. 模型训练 (与之前基本相同) ---
print("\n步骤 6: 开始训练 Wide&Deep 模型...")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None

for epoch in range(n_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # --- 验证步骤 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    avg_val_loss = val_loss / len(test_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, 验证损失: {avg_val_loss:.6f}")

    # --- 早停逻辑 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_weights = model.state_dict()
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"验证损失连续 {patience} 轮没有改善，触发早停！")
        break

if best_model_weights:
    model.load_state_dict(best_model_weights)
print("模型训练完成，已加载最佳模型权重。")

# --- 7. 模型保存 (保持不变) ---
model_filename = f'task1_Wide&Deep.pth'
torch.save(model.state_dict(), model_filename)
print(f"\n步骤 7: Wide&Deep 模型已保存到文件 -> {model_filename}")

# --- 8. 预测与评估 (保持不变) ---
print("\n步骤 8: 在测试集上进行预测和评估...")

model.eval()
with torch.no_grad():
    predictions_log = model(X_test_tensor).cpu().numpy().flatten()

y_test_original = np.expm1(y_test)
predictions_original = np.expm1(predictions_log)

mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

print("\n--- Wide&Deep 模型性能评估结果 ---")
print(f"平均绝对误差 (MAE): {mae:.2f} 小时")
print(f"均方根误差 (RMSE): {rmse:.2f} 小时")
print(f"R² 分数: {r2:.4f}")
print("------------------------")
print("\n说明:")
print(f"MAE: 模型的预测平均偏离真实PR处理时长约 {mae:.2f} 小时 (即约 {mae / 24:.2f} 天)。")
print(f"R² 分数: 模型可以解释测试集中PR处理时长变化的约 {r2 * 100:.2f}%。")