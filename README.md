## 🚆 台鐵不動產租金預測模型（Taiwan Railway Rent Predictor）

本專案建構一個針對台灣鐵路局不動產租賃資料的機器學習模型，預測各物件之月租金。透過資料清潔、特徵工程、CatBoost 演算法與 SHAP 特徵重要性解釋，打造穩定的租金回歸模型。

---

### 專案結構

```
taiwan_railway_rent_predictor/
├── taiwan_rent_data.json        # 原始資料（JSON 格式）
├── train_model.py               # 模型訓練與圖表輸出主程式
├── shap_summary_catboost_log_model.png  # SHAP 特徵重要性圖
├── 建物面積對每月租金(測試資料).png       # 回歸預測圖
├── 殘差分析圖(測試資料).png             # 殘差分佈圖
```

---

### 資料清潔與特徵工程

- 移除建物面積 ≤ 0 與月租金 > 30 萬的異常值
- 轉換欄位類型（建物面積、租金、日期）
- 提取「區名」與合併「經總度」
- 加入「剩餘租期（月）」作為特徵
- 使用初步 CatBoost 模型剖除高殘差資料（±2σ）

---

### 模型架構與設定

- 使用 [CatBoostRegressor](https://catboost.ai/)
- 設定如下：
  - `depth=5`
  - `l2_leaf_reg=10`
  - `learning_rate=0.05`
  - `iterations=1000`
  - `early_stopping_rounds=50`

---

### 模型效能評估

| 指標       | R² 分數 |
|------------|---------|
| 訓練集     | 0.1883  |
| 測試集     | 0.1419  |

- ✅ 模型穩定，無明顯過操合  
- ✅ 輸出對應圖表如下：

---

###  圖表輸出

#### 🔹 建物面積 vs 每月租金（測試資料）
![](./建物面積對每月租金(測試資料).png)

#### 🔹 殘差分析圖（測試資料）
![](./殘差分析圖(測試資料).png)

#### 🔹 特徵重要性分析（SHAP 值）
![](./特徵重要性分析(SHAP值).png)

---

###  未來優化方向

- 加入地理邻近性或房價指數作為輔助變數
- 擴增租期與交易歷史資料
- 實作多模型比較（如 XGBoost、LightGBM、Random Forest）

---

###  開發者

張中維（Chang, Chung-Wei）

