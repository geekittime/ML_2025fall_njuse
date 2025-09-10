import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge  # 修改点：导入线性回归
from sklearn.preprocessing import StandardScaler  # 修改点：导入标准化工具
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

# 忽略一些未来版本的警告，使输出更整洁
warnings.filterwarnings('ignore')

# --- 1. 数据加载与合并 ---
print("步骤 1: 开始加载和合并数据...")

# 使用脚本的绝对路径，使其不受运行位置影响
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
path_prefix = os.path.join(script_dir, 'yii2')

print(f"正在从以下路径加载数据: {path_prefix}")

pr_info_path = os.path.join(path_prefix, 'PR_info.xlsx')
pr_features_path = os.path.join(path_prefix, 'PR_features.xlsx')
author_features_path = os.path.join(path_prefix, 'author_features.xlsx')
pr_info_add_conversation_path = os.path.join(path_prefix, 'PR_info_add_conversation.xlsx')

pr_info = pd.read_excel(pr_info_path)
pr_features = pd.read_excel(pr_features_path)
author_features = pd.read_excel(author_features_path)
pr_info_add_conversation = pd.read_excel(pr_info_add_conversation_path)

# 遵照要求，保留此合并方式
merged_df = pd.merge(pr_info, pr_features, on='number', how='left')
merged_df = pd.merge(merged_df, author_features, on='number', how='left')
merged_df = pd.merge(merged_df, pr_info_add_conversation, on='number', how='left')

# 处理合并产生的同名列
if 'created_at_y' in merged_df.columns:
    merged_df.drop(columns=['created_at_y'], inplace=True)
if 'created_at_x' in merged_df.columns:
    merged_df.rename(columns={'created_at_x': 'created_at'}, inplace=True)

# 清理无用的索引列
columns_to_drop = [col for col in merged_df.columns if 'Unnamed: 0' in str(col)]
merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print(f"数据合并完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

# --- 2. 数据预处理与特征工程 ---
print("\n步骤 2: 开始数据预处理和特征工程...")

# 转换时间列为datetime对象
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
merged_df['closed_at'] = pd.to_datetime(merged_df['closed_at'])

# 计算目标变量 TTC
merged_df['TTC_hours'] = (merged_df['closed_at'] - merged_df['created_at']).dt.total_seconds() / 3600

# 过滤掉无效数据
merged_df.dropna(subset=['closed_at', 'created_at'], inplace=True)
merged_df = merged_df[merged_df['TTC_hours'] >= 0]
print(f"过滤掉无效数据后，剩余 {merged_df.shape[0]} 行。")

# 对目标变量进行log1p变换
merged_df['log_TTC_hours'] = np.log1p(merged_df['TTC_hours'])

# 选择用于训练的特征列
features = merged_df.select_dtypes(include=np.number).columns.tolist()

# 移除目标变量和ID等非特征列
features_to_remove = ['number', 'author_id', 'project_id', 'TTC_hours', 'log_TTC_hours']
features = [f for f in features if f not in features_to_remove]

X = merged_df[features]
y = merged_df['log_TTC_hours']

# 将无穷大值(inf)替换为NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 处理缺失值：使用中位数填充
print("处理缺失值：使用每列的中位数进行填充...")
X.fillna(X.median(), inplace=True)

print(f"最终选定的特征数量: {len(features)}")

# --- 3. 按时间顺序划分数据集 ---
print("\n步骤 3: 按时间顺序划分训练集和测试集...")
merged_df.sort_values('created_at', inplace=True)
X = X.loc[merged_df.index]
y = y.loc[merged_df.index]
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
print(f"训练集大小: {X_train.shape[0]} 行")
print(f"测试集大小: {X_test.shape[0]} 行")

# --- 修改点：为线性回归增加特征标准化步骤 ---
print("\n步骤 3.5: 为线性回归进行特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("特征标准化完成。")

# --- 4. 模型训练 ---
print("\n步骤 4: 开始训练线性回归模型...")

# --- 修改点：使用线性回归模型 ---
model = LinearRegression()
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

print("模型训练完成。")

# --- 5. 预测与评估 ---
print("\n步骤 5: 在测试集上进行预测和评估...")

# 使用标准化后的数据进行预测
predictions_log = model.predict(X_test_scaled)

# 将预测结果和真实值反转回原始尺度
y_test_original = np.expm1(y_test)
predictions_original = np.expm1(predictions_log)

# 计算评估指标
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

print("\n--- 模型性能评估结果 ---")
print(f"平均绝对误差 (MAE): {mae:.2f} 小时")
print(f"均方根误差 (RMSE): {rmse:.2f} 小时")
print(f"R² 分数: {r2:.4f}")
print("------------------------")
print("\n说明:")
print(f"MAE: 模型的预测平均偏离真实PR处理时长约 {mae:.2f} 小时 (即约 {mae / 24:.2f} 天)。")
print(f"R² 分数: 模型可以解释测试集中PR处理时长变化的约 {r2 * 100:.2f}%。")

# --- 修改点：分析线性回归的系数 ---
print("\n--- 线性回归模型系数分析 (Top 10 绝对值) ---")
# 将系数与特征名对应
coefficients = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_
})
# 计算系数的绝对值，用于排序
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
# 按绝对值降序排序
sorted_coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
print("系数解读：正数表示该特征增大时，处理时长倾向于增加；负数则相反。")
print(sorted_coefficients.head(10))
print("---------------------------------")