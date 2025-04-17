import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Tải dữ liệu Breast Cancer Wisconsin
data = load_breast_cancer()
X = data.data  # Đặc trưng (30 features)
y = data.target  # Nhãn (0: ác tính, 1: lành tính)
feature_names = data.feature_names

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Ridge Regression ---
ridge_model = Ridge()
ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid_search = GridSearchCV(ridge_model, ridge_param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train_scaled, y_train)

# Lấy mô hình Ridge tốt nhất
best_ridge = ridge_grid_search.best_estimator_
print(f"Best Ridge alpha: {ridge_grid_search.best_params_['alpha']}")

# Dự đoán với Ridge
ridge_y_pred_continuous = best_ridge.predict(X_test_scaled)
ridge_y_pred = (ridge_y_pred_continuous >= 0.5).astype(int)

# --- Lasso Regression ---
lasso_model = Lasso(max_iter=10000)
lasso_param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
lasso_grid_search = GridSearchCV(lasso_model, lasso_param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train_scaled, y_train)

# Lấy mô hình Lasso tốt nhất
best_lasso = lasso_grid_search.best_estimator_
print(f"Best Lasso alpha: {lasso_grid_search.best_params_['alpha']}")

# Dự đoán với Lasso
lasso_y_pred_continuous = best_lasso.predict(X_test_scaled)
lasso_y_pred = (lasso_y_pred_continuous >= 0.5).astype(int)

# --- Đánh giá mô hình ---
def evaluate_model(y_true, y_pred, y_pred_continuous, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred_continuous)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

# Đánh giá Ridge
evaluate_model(y_test, ridge_y_pred, ridge_y_pred_continuous, "Ridge Regression")

# Đánh giá Lasso
evaluate_model(y_test, lasso_y_pred, lasso_y_pred_continuous, "Lasso Regression")

# --- Phân tích đặc trưng được chọn bởi Lasso ---
print("\nLasso Feature Selection:")
selected_features = []
for feature, coef in zip(feature_names, best_lasso.coef_):
    if abs(coef) > 1e-5:
        selected_features.append((feature, coef))
        print(f"{feature}: {coef:.4f}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Number of eliminated features: {len(feature_names) - len(selected_features)}")

# --- Trực quan hóa trọng số của Ridge và Lasso ---
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(feature_names))

# Vẽ trọng số của Ridge
plt.bar(index, best_ridge.coef_, bar_width, label='Ridge Coefficients', color='skyblue')
# Vẽ trọng số của Lasso
plt.bar(index + bar_width, best_lasso.coef_, bar_width, label='Lasso Coefficients', color='salmon')

plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Coefficients: Ridge vs Lasso')
plt.xticks(index + bar_width / 2, feature_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("ridge_lasso_weights.png")
plt.close()

# --- Trực quan hóa dự đoán của Lasso ---
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Labels', alpha=0.6)
plt.scatter(range(len(y_test)), lasso_y_pred_continuous, color='red', label='Lasso Predictions', alpha=0.6)
plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Lasso Predictions vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.savefig("lasso_predictions.png")
plt.close()

print("\nVisualizations saved as 'ridge_lasso_weights.png' and 'lasso_predictions.png'")