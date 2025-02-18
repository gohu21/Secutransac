import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Chargement du dataset
df = pd.read_csv('transactions.csv')

# Suppression des doublons
df = df.drop_duplicates()

# Séparation features (X) et cible (y)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Prétraitement des colonnes
if 'timestamp' in X.columns:
    X['timestamp'] = pd.to_datetime(X['timestamp'])
    X['transaction_hour'] = X['timestamp'].dt.hour
    X['transaction_day'] = X['timestamp'].dt.dayofweek
    X['transaction_month'] = X['timestamp'].dt.month
    X['transaction_week'] = X['timestamp'].dt.isocalendar().week
    X['timestamp'] = X['timestamp'].view('int64') // 10**9  # Conversion en secondes UNIX

# Encodage des colonnes catégoriques
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

# Conversion des types pour éviter les erreurs avec SMOTE
X = X.astype(np.float64)

# Normalisation des données numériques
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Sauvegarde du StandardScaler pour `app.py`
joblib.dump(scaler, "scaler.pkl")

# Division train/test avec `random_state`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Rééchantillonnage avec SMOTE (éviter trop de fraudes)
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Ajuster pour éviter le suréchantillonnage
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Vérification après SMOTE
print("Distribution après SMOTE:")
print(y_train_resampled.value_counts())

# Initialisation du modèle XGBoost avec `random_state`
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=1,
    reg_alpha=2,
    reg_lambda=2,
    random_state=42
)

# Entraînement du modèle
xgb.fit(X_train_resampled, y_train_resampled)

# Sauvegarde du modèle entraîné
joblib.dump(xgb, 'best_xgboost_model.h5')

# Vérification des features importantes
plt.figure(figsize=(10, 6))
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Importance des Features dans XGBoost")
plt.show()

# Évaluation du modèle
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
