import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

from sklearn.preprocessing import StandardScaler

# 忽略一些未来版本的警告，使输出更整洁
warnings.filterwarnings('ignore')

# --- 1. 数据加载与合并 ---
print("步骤 1: 开始加载和合并数据...")

# 使用脚本的绝对路径，使其不受运行位置影响
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
path_prefix = os.path.join(script_dir, 'yii2')

print(f"正在从以下路径加载数据: {path_prefix}")

extracted_path = os.path.join(path_prefix, 'PR_extracted_features.xlsx')
merged_df = pd.read_excel(extracted_path)

print(f"数据合并完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")
print("合并后数据预览:\n", merged_df.head())

# --- 2. 数据预处理与特征工程 ---
print("\n步骤 2: 开始数据预处理和特征工程...")

# 转换时间列为datetime对象
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
# merged_df['closed_at'] = pd.to_datetime(merged_df['closed_at'])
merged_df['closed_at'] = pd.to_datetime(merged_df['closed_at'])
# 计算目标变量 TTC (Time-to-Close)，单位为小时
merged_df['TTC_hours'] = (merged_df['closed_at'] - merged_df['created_at']).dt.total_seconds() / 3600

# 过滤掉无效数据（例如，关闭时间早于创建时间或关闭时间为空）
merged_df.dropna(subset=['closed_at'], inplace=True)
merged_df = merged_df[merged_df['TTC_hours'] >= 0]
print(f"过滤掉无效数据后，剩余 {merged_df.shape[0]} 行。")
merged_df = merged_df[merged_df['TTC_hours'] <= 1000]
# 对目标变量进行log1p变换，以处理其长尾分布
merged_df['log_TTC_hours'] = np.log1p(merged_df['TTC_hours'])

# 同样对特征中的时长取log1p
merged_df['log_last_pr_update'] = np.log1p(merged_df['last_pr_update'])
merged_df['log_last_comment_update'] = np.log1p(merged_df['last_comment_update'])

# 将布尔列转换为数字0或1
for bool_col in merged_df.select_dtypes(include=bool).columns.tolist():
    merged_df[bool_col] = merged_df[bool_col].astype(int)

# 选择用于训练的特征列
features = merged_df.select_dtypes(include=np.number).columns.tolist()

# 从特征列表中移除目标变量和ID等非特征列
features_to_remove = ['number', 'TTC_hours', 'log_TTC_hours']
features = [f for f in features if f not in features_to_remove]

X = merged_df[features]
y = merged_df['log_TTC_hours']

# --- 【最终修正】: 在填充NaN之前，将无穷大值(inf)替换为NaN ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
# -----------------------------------------------------------

# 处理缺失值：使用中位数填充
print("处理缺失值：使用每列的中位数进行填充...")
X = X.fillna(X.median())

print(f"最终选定的特征数量: {len(features)}")

# --- 3. 按时间顺序划分数据集 ---
print("\n步骤 3: 按时间顺序划分训练集和测试集...")

# 确保数据按创建时间排序
merged_df.sort_values('created_at', inplace=True)
X = X.loc[merged_df.index]
y = y.loc[merged_df.index]

# 按80/20比例划分
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"训练集大小: {X_train.shape[0]} 行")
print(f"测试集大小: {X_test.shape[0]} 行")

# --- 修改点：增加特征标准化步骤 ---
print("\n步骤 3.5: 进行特征标准化...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("特征标准化完成。")


# --- 4. 模型训练 ---
print("\n步骤 4: 开始训练随机森林回归模型...")

model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
# 使用标准化后的数据进行训练
model.fit(X_train, y_train)

print("模型训练完成。")

# --- 5. 预测与评估 ---
print("\n步骤 5: 在测试集上进行预测和评估...")

log_predictions = model.predict(X_test)

# 将预测结果和真实值反转回原始尺度
y_test_original = np.expm1(y_test)
predictions_original = np.expm1(log_predictions)

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

# --- 特征重要性分析 ---
print("\n--- 特征重要性分析 (Top 10) ---")
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importances.head(10))
print("---------------------------------")
