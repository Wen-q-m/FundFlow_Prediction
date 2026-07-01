import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 第一步：读取所有数据（修复为读取真实 Excel）
# ==========================================
print("正在加载数据...")
df_balance = pd.read_excel('user_balance_table.xlsx')  # 核心交易数据
df_shibor = pd.read_csv('mfd_bank_shibor.csv')         # 银行间拆借利率
df_yield = pd.read_csv('mfd_day_share_interest.csv')   # 货币基金万份/七日年化

# ==========================================
# 第二步：数据清洗与聚合
# ==========================================
print("正在清洗与聚合数据...")
df_balance['report_date'] = pd.to_datetime(df_balance['report_date'], format='%Y%m%d')
daily_data = df_balance.groupby('report_date').agg({
    'total_purchase_amt': 'sum',
    'total_redeem_amt': 'sum',
}).reset_index().sort_values('report_date')

# 处理利率数据
if not df_shibor.empty:
    df_shibor['mfd_date'] = pd.to_datetime(df_shibor['mfd_date'], format='%Y%m%d')
    daily_data = pd.merge(daily_data, df_shibor, left_on='report_date', right_on='mfd_date', how='left')
    daily_data = daily_data.drop('mfd_date', axis=1)
if not df_yield.empty:
    df_yield['mfd_date'] = pd.to_datetime(df_yield['mfd_date'], format='%Y%m%d')
    daily_data = pd.merge(daily_data, df_yield, left_on='report_date', right_on='mfd_date', how='left')
    daily_data = daily_data.drop('mfd_date', axis=1)

# 修复报错：使用新版的 .ffill()
daily_data = daily_data.ffill().fillna(0)

# ==========================================
# 第三步：特征工程
# ==========================================
print("正在构建特征...")
daily_data['year'] = daily_data['report_date'].dt.year
daily_data['month'] = daily_data['report_date'].dt.month
daily_data['day_of_week'] = daily_data['report_date'].dt.dayofweek
daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)

for lag in [1, 3, 7]:
    daily_data[f'purchase_lag_{lag}'] = daily_data['total_purchase_amt'].shift(lag)
    daily_data[f'redeem_lag_{lag}'] = daily_data['total_redeem_amt'].shift(lag)

daily_data['purchase_ma_7'] = daily_data['total_purchase_amt'].rolling(window=7).mean()
daily_data['redeem_ma_7'] = daily_data['total_redeem_amt'].rolling(window=7).mean()
daily_data = daily_data.dropna().reset_index(drop=True)

# ==========================================
# 第四步：切分与训练
# ==========================================
print("正在切分数据集并开始训练...")
split_date = daily_data['report_date'].max() - pd.Timedelta(days=30)
train_df = daily_data[daily_data['report_date'] < split_date]
test_df = daily_data[daily_data['report_date'] >= split_date]

features = [col for col in daily_data.columns if col not in ['report_date', 'total_purchase_amt', 'total_redeem_amt']]
X_train, X_test = train_df[features], test_df[features]
y_train_p, y_test_p = train_df['total_purchase_amt'], test_df['total_purchase_amt']
y_train_r, y_test_r = train_df['total_redeem_amt'], test_df['total_redeem_amt']

# 训练模型
model_p = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_train, y_train_p)
model_r = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_train, y_train_r)

# 预测
pred_p = model_p.predict(X_test)
pred_r = model_r.predict(X_test)

# ==========================================
# 第五步：评估并画图
# ==========================================
rmse_p, mae_p = np.sqrt(mean_squared_error(y_test_p, pred_p)), mean_absolute_error(y_test_p, pred_p)
rmse_r, mae_r = np.sqrt(mean_squared_error(y_test_r, pred_r)), mean_absolute_error(y_test_r, pred_r)
r2_p = r2_score(y_test_p, pred_p)
r2_r = r2_score(y_test_r, pred_r)

print("\n============== 测试集评估结果（截图用） ==============")
print(f"申购预测 - RMSE: {rmse_p:.2f}, MAE: {mae_p:.2f}, R²: {r2_p:.4f}")
print(f"赎回预测 - RMSE: {rmse_r:.2f}, MAE: {mae_r:.2f}, R²: {r2_r:.4f}")

# 画图
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(test_df['report_date'], y_test_p, label='真实值(申购)')
plt.plot(test_df['report_date'], pred_p, label='预测值(申购)', linestyle='--')
plt.title('申购预测对比'); plt.legend(); plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
plt.plot(test_df['report_date'], y_test_r, label='真实值(赎回)')
plt.plot(test_df['report_date'], pred_r, label='预测值(赎回)', linestyle='--')
plt.title('赎回预测对比'); plt.legend(); plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# ==========================================
# 第六步：生成符合官方标准的提交文件（无表头纯数据）
# ==========================================
result_df = pd.DataFrame({
    'report_date': test_df['report_date'].dt.strftime('%Y%m%d'),
    'purchase': np.round(pred_p * 100).astype(int),
    'redeem': np.round(pred_r * 100).astype(int)
})

# 🔴 重点检查这一行最后的 header=False
result_df.to_csv('tc_comp_predict_table.csv', index=False, header=False) 
print("\n✅ 已生成符合官方要求的提交文件: tc_comp_predict_table.csv")