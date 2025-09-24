import argparse
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
# --- 修改点: 为回归任务新增评估指标 ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, \
    mean_absolute_error

# --- 导入 PyTorch 相关库 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

# 忽略一些未来版本的警告
warnings.filterwarnings('ignore')


# --- 新增：定义多任务 Wide&Deep 模型 ---
class MultiTaskWideAndDeep(nn.Module):
    def __init__(self, input_dim, deep_dims=[128, 64, 32]):
        super(MultiTaskWideAndDeep, self).__init__()

        # --- 共享的 Deep 部分 (与原模型一致) ---
        deep_layers = []
        current_dim = input_dim
        for h_dim in deep_dims:
            deep_layers.append(nn.Linear(current_dim, h_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.2))
            current_dim = h_dim
        self.deep_part = nn.Sequential(*deep_layers)

        # 拼接 Wide 部分 (原始输入) 和 Deep 部分的最终输出
        combined_input_dim = input_dim + deep_dims[-1]

        # --- 任务一：分类头 (Classification Head) ---
        self.classification_head = nn.Sequential(
            nn.Linear(combined_input_dim, 1),
            nn.Sigmoid()
        )

        # --- 任务二：回归头 (Regression Head) ---
        self.regression_head = nn.Sequential(
            nn.Linear(combined_input_dim, 1)
        )

    def forward(self, x):
        # 共享的Deep部分的输出
        deep_output = self.deep_part(x)
        # 拼接 Wide 部分 (原始输入x) 和 Deep 部分的输出
        combined_input = torch.cat([x, deep_output], dim=1)
        # 得到两个任务的输出
        classification_output = self.classification_head(combined_input)
        regression_output = self.regression_head(combined_input)
        return classification_output, regression_output



def prepare_multi_task_data(project_name="yii2"):
    """加载和预处理数据，为两个任务返回标签"""
    print("步骤 1: 开始加载和合并数据...")
    # --- 以下数据加载和路径逻辑，完全遵照您的原始代码 ---
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    # parent_dir = os.path.dirname(script_dir)
    path_prefix = os.path.join(script_dir, project_name)

    extracted_path = os.path.join(path_prefix, 'PR_extracted_features.xlsx')
    merged_df = pd.read_excel(extracted_path)

    print(f"数据加载完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

    print("\n步骤 2: 开始为多任务进行数据预处理和特征工程...")

    for bool_col in merged_df.select_dtypes(include=bool).columns.tolist():
        merged_df[bool_col] = merged_df[bool_col].astype(int)

    merged_df['TTC_hours'] = (merged_df['closed_at'] - merged_df['created_at']).dt.total_seconds() / 3600
    merged_df = merged_df[merged_df['TTC_hours'] <= 1000]
    merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
    merged_df['closed_at'] = pd.to_datetime(merged_df['closed_at'])
    merged_df.dropna(subset=['closed_at', 'created_at'], inplace=True)
    merged_df = merged_df[(merged_df['TTC_hours'] >= 0) & (merged_df['TTC_hours'] <= 1000)]
    merged_df['log_TTC_hours'] = np.log1p(merged_df['TTC_hours'])
    merged_df['log_last_pr_update'] = np.log1p(merged_df['last_pr_update'])
    merged_df['log_last_comment_update'] = np.log1p(merged_df['last_comment_update'])

    # --- 修改点: 同时提取分类和回归两个任务的标签 ---
    y_class = merged_df['merged']
    y_reg = merged_df['log_TTC_hours']  # 新增回归任务标签

    features = merged_df.select_dtypes(include=np.number).columns.tolist()
    features_to_remove = [
        'number', 'merged', 'closed_at', 'TTC_hours', 'log_TTC_hours',
        'last_pr_update', 'last_comment_update', 'log_last_pr_update', 'log_last_comment_update'
    ]
    features = [f for f in features if f not in features_to_remove]
    X = merged_df[features]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # --- 新增: 对回归标签也进行数据清洗 ---
    y_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_reg.fillna(y_reg.median(), inplace=True)

    print(f"最终选定的特征数量: {len(features)}")

    print("\n步骤 3: 按时间顺序划分与标准化...")
    # --- 以下划分逻辑，完全遵照您的原始代码 ---
    merged_df.sort_values('created_at', inplace=True)
    X = X.loc[merged_df.index]
    y_class = y_class.loc[merged_df.index]  # 修改点: 重命名为 y_class
    y_reg = y_reg.loc[merged_df.index]  # 新增

    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]

    # 修改点: 划分两个任务的标签
    y_train_class, y_test_class = y_class.iloc[:split_point], y_class.iloc[split_point:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_point], y_reg.iloc[split_point:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 修改点: 返回两个任务的数据
    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_reg, y_test_reg, len(features)


# --- 新增：训练多任务模型 ---
def train_multi_task_model(X_train_scaled, y_train_class, y_train_reg, X_test_scaled, y_test_class, y_test_reg,
                           input_dim, device):
    """训练一个多任务模型"""
    X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
    y_train_class_tensor = torch.tensor(y_train_class.values.astype(np.float32)).to(device).view(-1, 1)
    y_train_reg_tensor = torch.tensor(y_train_reg.values.astype(np.float32)).to(device).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
    y_test_class_tensor = torch.tensor(y_test_class.values.astype(np.float32)).to(device).view(-1, 1)
    y_test_reg_tensor = torch.tensor(y_test_reg.values.astype(np.float32)).to(device).view(-1, 1)

    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_class_tensor, y_train_reg_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_class_tensor, y_test_reg_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskWideAndDeep(input_dim).to(device)

    criterion_class = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    for epoch in range(n_epochs):
        model.train()
        for inputs, labels_class, labels_reg in train_loader:
            optimizer.zero_grad()
            outputs_class, outputs_reg = model(inputs)
            loss_class = criterion_class(outputs_class, labels_class)
            loss_reg = criterion_reg(outputs_reg, labels_reg)
            total_loss = loss_class + loss_reg  # 核心：将两个loss相加
            total_loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels_class, labels_reg in test_loader:
                outputs_class, outputs_reg = model(inputs)
                loss_class = criterion_class(outputs_class, labels_class)
                loss_reg = criterion_reg(outputs_reg, labels_reg)
                val_loss += (loss_class.item() + loss_reg.item()) * inputs.size(0)

        avg_val_loss = val_loss / len(test_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, 验证总损失: {avg_val_loss:.6f}")

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

    print("多任务模型训练完成。")
    model_filename = 'multitask_model.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"\n模型已保存到 -> {model_filename}")
    return model


# --- 新增：评估多任务模型 ---
def eval_multi_task_model(model, X_test_scaled, y_test_class, y_test_reg, device):
    """分别评估多任务模型在两个任务上的表现"""
    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        pred_class_probs, pred_reg_values = model(X_test_tensor)
        pred_class_labels = (pred_class_probs.cpu().numpy().flatten() > 0.5).astype(int)
        pred_reg_values = pred_reg_values.cpu().numpy().flatten()
        pred_ttc_hours = np.expm1(pred_reg_values)
        y_test_hours=np.expm1(y_test_reg)

    print("\n\n=============== 任务一：PR合并预测 (分类) 评估 ================")
    print("\n--- 混淆矩阵 ---")
    print(confusion_matrix(y_test_class, pred_class_labels))
    print("\n--- 分类报告 ---")
    print(classification_report(y_test_class, pred_class_labels, target_names=['Not Merged (0)', 'Merged (1)']))

    print("\n\n========== 任务二：PR关闭时间预测 (回归) 评估 ==========")
    print(f"均方误差 (MSE): {mean_squared_error(pred_ttc_hours, y_test_hours):.4f}")
    print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test_hours, pred_ttc_hours):.4f}")
    print("---------------------------------------------------------")


# --- 修改点：主函数，调用新的多任务函数 ---
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用PyTorch多任务Wide&Deep模型完成PR预测任务")
    parser.add_argument("--do_train", action="store_true", help="训练一个新模型")
    parser.add_argument("--do_eval", action="store_true", help="评估一个已存在的模型")
    parser.add_argument("--project_name", type=str, default="yii2", help="要使用的数据项目文件夹名")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 调用新的数据准备函数
    X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_reg, y_test_reg, input_dim = prepare_multi_task_data(
        args.project_name)

    model = None
    if args.do_train:
        # 调用新的训练函数
        model = train_multi_task_model(X_train_scaled, y_train_class, y_train_reg, X_test_scaled, y_test_class,
                                       y_test_reg, input_dim, device)

    if args.do_eval:
        if model is None:
            # 加载新的多任务模型
            model = MultiTaskWideAndDeep(input_dim).to(device)
            model_path = 'multitask_model.pth'
            print(f"\n正在从 {model_path} 加载已训练的模型...")
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except FileNotFoundError:
                print(f"错误：找不到已保存的模型文件 {model_path}。请先使用 --do_train 进行训练。")
                return
        # 调用新的评估函数
        eval_multi_task_model(model, X_test_scaled, y_test_class, y_test_reg, device)


if __name__ == "__main__":
    main()