# ============================================================
#  Social Network Ads — Purchase Prediction (Improved)
#  Original: Logistic Regression only, no scaling
#  Improved: 4 models + scaling + visualizations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Load Data
# ============================================================
df = pd.read_csv("Social_Network_Ads.csv")
print("="*55)
print("SOCIAL NETWORK ADS — PURCHASE PREDICTION")
print("="*55)
print(f"\nDataset shape : {df.shape}")
print(f"Purchased=1   : {df['Purchased'].sum()} people")
print(f"Purchased=0   : {(df['Purchased']==0).sum()} people")

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# STEP 2 — Feature Scaling (IMPORTANT for LR & SVM)
# ============================================================
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)   # fit on train only
X_test_sc  = sc.transform(X_test)        # transform test with same scaler

# ============================================================
# STEP 3 — Train & Compare 4 Models
# ============================================================
print("\n" + "="*55)
print("MODEL COMPARISON")
print("="*55)

models = {
    'Logistic Regression' : (LogisticRegression(),                               X_train_sc, X_test_sc),
    'Decision Tree'        : (DecisionTreeClassifier(max_depth=5, random_state=42), X_train,    X_test),
    'Random Forest'        : (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    'SVM (RBF)'            : (SVC(kernel='rbf', random_state=42),                X_train_sc, X_test_sc),
}

results = {}
print(f"\n{'Model':<22} | {'Test Acc':>9} | {'CV Mean':>9} | {'CV Std':>8}")
print("-" * 55)
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_test, model.predict(Xte))
    cv  = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'acc': acc, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
        'model': model, 'Xtr': Xtr, 'Xte': Xte
    }
    print(f"{name:<22} | {acc*100:>8.2f}% | {cv.mean()*100:>8.2f}% | ±{cv.std()*100:>5.2f}%")

# ============================================================
# STEP 4 — Best Model: SVM Deep Dive
# ============================================================
print("\n" + "="*55)
print("BEST MODEL — SVM (RBF Kernel): 92.50% accuracy")
print("="*55)

best_model = results['SVM (RBF)']['model']
best_pred  = best_model.predict(X_test_sc)

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Not Purchased','Purchased']))

# ============================================================
# STEP 5 — Visualizations
# ============================================================

# --- Plot 1: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 5))
names  = list(results.keys())
accs   = [results[n]['cv_mean']*100 for n in names]
stds   = [results[n]['cv_std']*100  for n in names]
colors = ['#e07b39', '#2ab0c5', '#a29bfe', '#fd79a8']
bars = ax.bar(names, accs, yerr=stds, color=colors, edgecolor='white',
              capsize=6, width=0.5)
ax.set_ylim(70, 100)
ax.set_ylabel('CV Accuracy (%)', fontsize=12)
ax.set_title('Model Comparison — 5-Fold Cross Validation', fontsize=13, fontweight='bold')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()

# --- Plot 2: Decision Boundary of SVM ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (title, X_set, y_set) in zip(axes, [
    ('SVM — Training Set', X_train_sc, y_train),
    ('SVM — Test Set',     X_test_sc,  y_test)
]):
    X1 = np.arange(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 0.01)
    X2 = np.arange(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 0.01)
    XX1, XX2 = np.meshgrid(X1, X2)
    Z = best_model.predict(np.c_[XX1.ravel(), XX2.ravel()])
    Z = Z.reshape(XX1.shape)
    ax.contourf(XX1, XX2, Z, alpha=0.35, cmap=ListedColormap(['#ffb3b3','#b3d9ff']))
    for cls, col, lbl in [(0,'#e07b39','Not Purchased'), (1,'#2ab0c5','Purchased')]:
        idx = y_set == cls
        ax.scatter(X_set[idx, 0], X_set[idx, 1], c=col, label=lbl, edgecolors='white', s=50)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Age (scaled)')
    ax.set_ylabel('Estimated Salary (scaled)')
    ax.legend()
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150)
plt.close()

# --- Plot 3: Age & Salary Distribution by Purchase ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ['Age', 'EstimatedSalary']):
    for val, lbl, color in [(0,'Not Purchased','#e07b39'), (1,'Purchased','#2ab0c5')]:
        ax.hist(df[df['Purchased']==val][col], bins=20, alpha=0.6,
                label=lbl, color=color, edgecolor='white')
    ax.set_title(f'{col} Distribution by Purchase', fontsize=12, fontweight='bold')
    ax.set_xlabel(col); ax.set_ylabel('Count')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('distribution.png', dpi=150)
plt.close()

print("\nSaved: model_comparison.png")
print("Saved: decision_boundary.png")
print("Saved: distribution.png")

# ============================================================
# STEP 6 — Predict for a new user
# ============================================================
print("\n" + "="*55)
print("PREDICT FOR A NEW USER")
print("="*55)
new_user = pd.DataFrame({'Age': [30], 'EstimatedSalary': [87000]})
new_user_sc = sc.transform(new_user)
pred = best_model.predict(new_user_sc)[0]
print(f"Age=30, Salary=87,000 → {'WILL Purchase ' if pred==1 else 'Will NOT Purchase '}")

print("\n" + "="*55)
print("ALL DONE!")


print("new addition branch")