import argparse
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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


# --- 修改点：定义一个用于二分类的神经网络 ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 分类任务，输出层1个神经元
            nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出转换为0-1之间的概率
        )

    def forward(self, x):
        return self.layers(x)


def prepare_classification_data(project_name="yii2"):
    """加载和预处理数据，为分类任务返回训练/测试数据和特征维度"""
    print("步骤 1: 开始加载和合并数据...")

    # 数据加载部分保持不变
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(script_dir)
    path_prefix = os.path.join(parent_dir, project_name)

    extracted_path = os.path.join(path_prefix, 'PR_extracted_features.xlsx')
    merged_df = pd.read_excel(extracted_path)
    print(f"数据加载完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

    # --- 修改点：针对分类任务的预处理 ---
    print("\n步骤 2: 开始为分类任务进行数据预处理和特征工程...")

    # 转换布尔列和时间列
    for bool_col in merged_df.select_dtypes(include=bool).columns.tolist():
        merged_df[bool_col] = merged_df[bool_col].astype(int)
    merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])

    # 定义目标变量 y
    y = merged_df['merged']

    # 选择特征
    features = merged_df.select_dtypes(include=np.number).columns.tolist()

    # 移除目标变量和ID，以及所有与“未来”信息相关的列，防止数据泄漏
    features_to_remove = [
        'number', 'merged', 'closed_at', 'TTC_hours', 'log_TTC_hours',
        'last_pr_update', 'last_comment_update', 'log_last_pr_update', 'log_last_comment_update'
    ]
    features = [f for f in features if f not in features_to_remove]

    X = merged_df[features]

    # 清理无穷大值并填充缺失值
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    print(f"最终选定的特征数量: {len(features)}")
    print("使用的特征:", features)

    # --- 按时间顺序划分 ---
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

    return X_train_scaled, X_test_scaled, y_train, y_test, len(features)


def train_classifier(X_train_scaled, y_train, X_test_scaled, y_test, input_dim, project_name, device):
    """训练一个分类模型"""
    # 转 tensor
    X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).to(device).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
    y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).to(device).view(-1, 1)

    # DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    model = MLPClassifier(input_dim).to(device)

    # 训练
    # --- 修改点：使用二元交叉熵损失函数 ---
    criterion = nn.BCELoss()
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

        # 验证
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

    print("模型训练完成。")
    model_filename = f'task2_MLP.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"\n模型已保存到 -> {model_filename}")
    return model


def eval_classifier(model, X_test_scaled, y_test, device):
    """评估一个分类模型"""
    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        # 模型的输出是概率
        probabilities = model(X_test_tensor).cpu().numpy().flatten()
        # 以0.5为阈值，将概率转换为类别 (0或1)
        predictions = (probabilities > 0.5).astype(int)

    print("\n--- 混淆矩阵 ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print("--------------------")

    print(f"\n总预测正确率 (Accuracy): {accuracy_score(y_test, predictions):.4f}")

    print("\n--- 分类报告 ---")
    report = classification_report(y_test, predictions, target_names=['Not Merged (0)', 'Merged (1)'])
    print(report)
    print("--------------------")


def main():
    parser = argparse.ArgumentParser(description="使用PyTorch神经网络完成PR分类任务")
    parser.add_argument("--do_train", action="store_true", help="训练一个新模型")
    parser.add_argument("--do_eval", action="store_true", help="评估一个已存在的模型")
    parser.add_argument("--project_name", type=str, default="yii2", help="要使用的数据项目文件夹名")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_scaled, X_test_scaled, y_train, y_test, input_dim = prepare_classification_data(args.project_name)

    model = None
    if args.do_train:
        model = train_classifier(X_train_scaled, y_train, X_test_scaled, y_test, input_dim, args.project_name, device)

    if args.do_eval:
        if model is None:  # 如果没有训练，就直接加载已保存的模型
            model = MLPClassifier(input_dim).to(device)
            model_path = 'task2_MLP.pth'
            print(f"\n正在从 {model_path} 加载已训练的模型...")
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except FileNotFoundError:
                print(f"错误：找不到已保存的模型文件 {model_path}。请先使用 --do_train 进行训练。")
                return
        eval_classifier(model, X_test_scaled, y_test, device)


if __name__ == "__main__":
    main()