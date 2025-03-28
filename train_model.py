import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import json

# ✅ 本地 JSON 路徑
file_path = "C:/Users/user/Desktop/taiwan_railway_rent_predictor/taiwan_rent_data.json"

with open(file_path, "r", encoding="utf-8-sig") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 數值處理
df["建物面積"] = pd.to_numeric(df["建物面積"], errors="coerce")
df["每月租金"] = pd.to_numeric(df["每月租金"], errors="coerce")
df["總樓層數"] = pd.to_numeric(df["總樓層數"], errors="coerce")
df["租期屆滿"] = pd.to_datetime(df["租期屆滿"], errors="coerce")

# 初步清洗
df = df[["縣市", "實際用途", "建物面積", "構造", "總樓層數", "建物現況", "房屋座落", "租期屆滿", "每月租金"]]
df = df.dropna(subset=["建物面積", "每月租金"])
df = df[df["建物面積"] > 0]
df = df[df["每月租金"] < 300000]

# ➕ 剩餘租期（月）
today = pd.Timestamp.today()
df["剩餘租期(月)"] = (df["租期屆滿"] - today).dt.days // 30
df["剩餘租期(月)"] = df["剩餘租期(月)"].fillna(0).astype(int)

# ➕ 區名
df["區名"] = df["房屋座落"].str.extract(r"(..區)")
df["區名"] = df["區名"].fillna("未知區")

# ➕ 經緯度內建快取
latlon_cache = pd.DataFrame([
    ["台北市", "大安區", 25.026, 121.543],
    ["台北市", "信義區", 25.033, 121.562],
    ["台北市", "中山區", 25.063, 121.522],
    ["新北市", "板橋區", 25.013, 121.464],
    ["新北市", "新店區", 24.959, 121.538],
    ["桃園市", "中壢區", 24.957, 121.226],
    ["台中市", "西屯區", 24.181, 120.641],
    ["台中市", "北區", 24.157, 120.685],
    ["台南市", "東區", 22.993, 120.223],
    ["高雄市", "苓雅區", 22.626, 120.311],
    ["高雄市", "左營區", 22.686, 120.293],
    ["花蓮縣", "花蓮市", 23.973, 121.601],
    ["台東縣", "台東市", 22.755, 121.144],
    ["宜蘭縣", "宜蘭市", 24.756, 121.754],
    ["基隆市", "仁愛區", 25.129, 121.741],
], columns=["縣市", "區名", "lat", "lon"])

df = df.merge(latlon_cache, on=["縣市", "區名"], how="left")

# ➕ log(租金)
df["log_租金"] = np.log1p(df["每月租金"])

# 預篩模型以剔除高殘差
X_raw = df[["縣市", "實際用途", "建物面積", "構造", "總樓層數", "建物現況", "區名", "剩餘租期(月)", "lat", "lon"]]
y_raw = df["log_租金"]
cat_features = ["縣市", "實際用途", "構造", "建物現況", "區名"]

model_init = CatBoostRegressor(verbose=0, random_state=42)
model_init.fit(X_raw, y_raw, cat_features=cat_features)
y_pred_raw = model_init.predict(X_raw)
residuals_raw = y_raw - y_pred_raw
std = residuals_raw.std()

df["殘差"] = residuals_raw
df_filtered = df[np.abs(df["殘差"]) <= 2 * std].copy()
print("🧹 篩除高殘差樣本後剩餘筆數：", len(df_filtered))

# 正式資料
X = df_filtered[["縣市", "實際用途", "建物面積", "構造", "總樓層數", "建物現況", "區名", "剩餘租期(月)", "lat", "lon"]]
y = df_filtered["log_租金"]

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_, X_valid, y_train_, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ✅ CatBoost 模型（防過擬合參數）
model = CatBoostRegressor(
    depth=5,
    l2_leaf_reg=10,
    learning_rate=0.05,
    iterations=1000,
    early_stopping_rounds=50,
    random_state=42,
    verbose=100
)

model.fit(
    X_train_,
    y_train_,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features,
    use_best_model=True
)

# 預測與還原
y_train_pred = np.expm1(model.predict(X_train))
y_test_pred = np.expm1(model.predict(X_test))
y_train_true = np.expm1(y_train)
y_test_true = np.expm1(y_test)

# 評估
r2_train = r2_score(y_train_true, y_train_pred)
r2_test = r2_score(y_test_true, y_test_pred)
print("✅ 訓練 R²:", round(r2_train, 4))
print("✅ 測試 R² :", round(r2_test, 4))
if r2_train - r2_test > 0.1:
    print("⚠️ 模型可能仍有過擬合")
else:
    print("✅ 模型表現穩定，沒有明顯過擬合")

# 回歸圖
df_vis = X_test.copy()
df_vis["每月租金"] = y_test_true
df_vis["預測租金"] = y_test_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x="建物面積", y="每月租金", data=df_vis, label="實際值")
sns.lineplot(x="建物面積", y="預測租金", data=df_vis, color="red", label="預測值")
plt.title("建物面積 vs 每月租金（測試集回歸圖）")
plt.xlabel("建物面積")
plt.ylabel("每月租金")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 殘差圖
residuals = df_vis["每月租金"] - df_vis["預測租金"]
plt.figure(figsize=(10, 6))
sns.residplot(x=df_vis["建物面積"], y=residuals, lowess=True, color="purple")
plt.title("殘差圖（測試集）")
plt.xlabel("建物面積")
plt.ylabel("殘差")
plt.grid(True)
plt.tight_layout()
plt.show()

# SHAP 圖表
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("C:/Users/user/Desktop/taiwan_railway_rent_predictor/shap_summary_catboost_log_model.png")
plt.show()