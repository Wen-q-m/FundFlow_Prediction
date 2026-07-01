import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 第一步：读取所有数据
# ==========================================
# 请确保这4个csv文件在代码同一目录下，或者修改为你的绝对路径
df_balance = pd.read_csv('user_balance_table.csv')  # 核心交易数据
df_shibor = pd.read_csv('mfd_bank_shibor.csv')      # 银行间拆借利率
df_yield = pd.read_csv('mfd_day_share_interest.csv') # 货币基金万份/七日年化
# 用户画像表本次作为宏观统计特征使用（先忽略单用户画像，聚合成日维度的宏观特征）

# ==========================================
# 第二步：数据清洗与聚合（非常关键）
# ==========================================
# 1. 处理日期格式，统一转为 datetime
df_balance['report_date'] = pd.to_datetime(df_balance['report_date'], format='%Y%m%d')
df_shibor['mfd_date'] = pd.to_datetime(df_shibor['mfd_date'], format='%Y%m%d')
df_yield['mfd_date'] = pd.to_datetime(df_yield['mfd_date'], format='%Y%m%d')

# 2. 将用户每日交易表按日期聚合（目标是预测每天的总申购和总赎回）
daily_data = df_balance.groupby('report_date').agg({
    'total_purchase_amt': 'sum',  # 每天总申购
    'total_redeem_amt': 'sum',    # 每天总赎回
}).reset_index().sort_values('report_date')

# 3. 合并拆借利率表（以日期为基准左连接）
daily_data = pd.merge(daily_data, df_shibor, left_on='report_date', right_on='mfd_date', how='left')
daily_data = daily_data.drop('mfd_date', axis=1)

# 4. 合并万份/七日年化收益率表
daily_data = pd.merge(daily_data, df_yield, left_on='report_date', right_on='mfd_date', how='left')
daily_data = daily_data.drop('mfd_date', axis=1)

# 5. 填充缺失的利率数据（非交易日无利率，用前值填充）
daily_data = daily_data.fillna(method='ffill')

# ==========================================
# 第三步：特征工程（时间序列专属）
# ==========================================
# 1. 基础日期特征
daily_data['year'] = daily_data['report_date'].dt.year
daily_data['month'] = daily_data['report_date'].dt.month
daily_data['day'] = daily_data['report_date'].dt.day
daily_data['day_of_week'] = daily_data['report_date'].dt.dayofweek  # 0=周一，6=周日
daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)

# 2. 滞后特征 (Lag) - 资金流动具有“惯性”
for lag in [1, 3, 7]:
    daily_data[f'purchase_lag_{lag}'] = daily_data['total_purchase_amt'].shift(lag)
    daily_data[f'redeem_lag_{lag}'] = daily_data['total_redeem_amt'].shift(lag)

# 3. 滚动统计特征 (Rolling) - 历史趋势
daily_data['purchase_ma_7'] = daily_data['total_purchase_amt'].rolling(window=7).mean()
daily_data['redeem_ma_7'] = daily_data['total_redeem_amt'].rolling(window=7).mean()
daily_data['purchase_std_7'] = daily_data['total_purchase_amt'].rolling(window=7).std()
daily_data['redeem_std_7'] = daily_data['total_redeem_amt'].rolling(window=7).std()

# 4. 删除因为滞后/滚动产生的空值行（前7天没有历史数据，丢掉）
daily_data = daily_data.dropna().reset_index(drop=True)

# ==========================================
# 第四步：划分训练集与测试集（时间顺序切分，严禁随机乱切）
# ==========================================
# 取最后 30 天作为测试集
split_date = daily_data['report_date'].max() - pd.Timedelta(days=30)
train_df = daily_data[daily_data['report_date'] < split_date]
test_df = daily_data[daily_data['report_date'] >= split_date]

# 定义特征列（排除日期和两个目标变量）
features = [col for col in daily_data.columns if col not in ['report_date', 'total_purchase_amt', 'total_redeem_amt']]

X_train = train_df[features]
y_train_purchase = train_df['total_purchase_amt']
y_train_redeem = train_df['total_redeem_amt']

X_test = test_df[features]
y_test_purchase = test_df['total_purchase_amt']
y_test_redeem = test_df['total_redeem_amt']

# ==========================================
# 第五步：训练 LightGBM 模型（传统机器学习，符合课设要求）
# ==========================================
print("开始训练申购预测模型...")
model_purchase = lgb.LGBMRegressor(
    objective='regression', 
    num_leaves=31, 
    learning_rate=0.05, 
    n_estimators=500,
    random_state=42,
    verbose=-1
)
model_purchase.fit(X_train, y_train_purchase)

print("开始训练赎回预测模型...")
model_redeem = lgb.LGBMRegressor(
    objective='regression', 
    num_leaves=31, 
    learning_rate=0.05, 
    n_estimators=500,
    random_state=42,
    verbose=-1
)
model_redeem.fit(X_train, y_train_redeem)

# ==========================================
# 第六步：预测、评估与画图（生成截图用）
# ==========================================
# 预测
pred_purchase = model_purchase.predict(X_test)
pred_redeem = model_redeem.predict(X_test)

# 计算指标：RMSE和MAE（截图用）
rmse_purchase = np.sqrt(mean_squared_error(y_test_purchase, pred_purchase))
rmse_redeem = np.sqrt(mean_squared_error(y_test_redeem, pred_redeem))
mae_purchase = mean_absolute_error(y_test_purchase, pred_purchase)
mae_redeem = mean_absolute_error(y_test_redeem, pred_redeem)

print("\n============== 测试集评估结果 ==============")
print(f"申购预测 - RMSE: {rmse_purchase:.2f}, MAE: {mae_purchase:.2f}")
print(f"赎回预测 - RMSE: {rmse_redeem:.2f}, MAE: {mae_redeem:.2f}")

# 绘制对比图（截这张图放到报告里）
plt.figure(figsize=(14, 6))

# 子图1：申购
plt.subplot(1, 2, 1)
plt.plot(test_df['report_date'], y_test_purchase, label='真实值(申购)', color='blue')
plt.plot(test_df['report_date'], pred_purchase, label='预测值(申购)', color='red', linestyle='--')
plt.title(f'申购预测对比 (RMSE: {rmse_purchase:.0f})')
plt.xlabel('日期')
plt.ylabel('金额')
plt.legend()
plt.xticks(rotation=45)

# 子图2：赎回
plt.subplot(1, 2, 2)
plt.plot(test_df['report_date'], y_test_redeem, label='真实值(赎回)', color='green')
plt.plot(test_df['report_date'], pred_redeem, label='预测值(赎回)', color='orange', linestyle='--')
plt.title(f'赎回预测对比 (RMSE: {rmse_redeem:.0f})')
plt.xlabel('日期')
plt.ylabel('金额')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()  # 请在运行结束后，用截图工具截取这张图，贴进报告