"""
automl_single.py  —  تطبيق No-Code ML في ملف واحد (نسخة Pro v2)
التشغيل:  streamlit run automl_single.py
اختياري:  pip install imbalanced-learn shap xgboost openpyxl

الصفحات: البيانات ← التحليل ← التنضيف (Outliers) ← التجهيز ← الخوارزميات ← النتائج ← الكود
بيدعم: Classification / Regression / Clustering / Anomaly Detection / PCA-tSNE / Neural Network (MLP)

اللي اتصلّح في النسخة دي:
  • Data Leakage: كل التجهيز (Imputation / Scaling / Encoding / Target Encoding / SMOTE) جوه Pipeline
    بيتعمله fit على الـ Train بس، والـ Outliers بتتعالج بعد الـ split.
  • مراجعة أنواع الأعمدة يدويًا (auto / numeric / categorical / drop) + استبعاد الـ IDs أوتوماتيك.
  • Missing: Median/Mean/Constant + Missing-Indicator أوتوماتيك لو الـ Missing كتير.
  • Imbalanced: المقياس الافتراضي F1 / PR-AUC + class_weight أو SMOTE.
  • High Cardinality: Target Encoding بدل One-Hot.
  • حدود للموارد: أقصى عدد صفوف للتدريب + تحذير للخوارزميات التقيلة + CV اختياري.
  • Explainability: Permutation Importance (+ SHAP للموديلات الشجرية لو متسطّب).
"""
import base64
import io
import json
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

warnings.filterwarnings("ignore")

UPLOAD_DIR = "uploaded_datasets"   # الملفات المرفوعة
MODELS_DIR = "saved_models"        # الموديلات المحفوظة
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== ALGORITHMS (ضيف خوارزمية جديدة هنا بسطر واحد) ====================
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import OneClassSVM, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# صيغة الـ params: ("int", min, max, default) | ("float", min, max, default)
#                  ("choice", [options], default) | ("layers", default_text)  مثال "64,32"
_MLP_PARAMS = {
    "hidden_layer_sizes": ("layers", "64,32"),
    "activation": ("choice", ["relu", "tanh", "logistic"], "relu"),
    "alpha": ("float", 0.0, 1.0, 0.0001),
    "learning_rate_init": ("float", 0.0001, 0.1, 0.001),
    "max_iter": ("int", 100, 2000, 500),
}
_MLP_FIXED = {"random_state": 42, "early_stopping": True}

CLASSIFIERS = {
    "Logistic Regression": {"cls": LogisticRegression, "fixed": {"max_iter": 2000},
                            "params": {"C": ("float", 0.01, 20.0, 1.0)}},
    "KNN": {"cls": KNeighborsClassifier, "fixed": {}, "params": {"n_neighbors": ("int", 1, 30, 5)}},
    "SVM": {"cls": SVC, "fixed": {}, "params": {"C": ("float", 0.01, 20.0, 1.0),
                                                 "kernel": ("choice", ["rbf", "linear", "poly"], "rbf")}},
    "Decision Tree": {"cls": DecisionTreeClassifier, "fixed": {"random_state": 42},
                      "params": {"max_depth": ("int", 1, 30, 5)}},
    "Random Forest": {"cls": RandomForestClassifier, "fixed": {"random_state": 42, "n_jobs": -1},
                      "params": {"n_estimators": ("int", 10, 500, 100), "max_depth": ("int", 1, 30, 10)}},
    "Extra Trees": {"cls": ExtraTreesClassifier, "fixed": {"random_state": 42, "n_jobs": -1},
                    "params": {"n_estimators": ("int", 10, 500, 100), "max_depth": ("int", 1, 30, 10)}},
    "Gradient Boosting": {"cls": GradientBoostingClassifier, "fixed": {"random_state": 42},
                          "params": {"n_estimators": ("int", 10, 500, 100), "learning_rate": ("float", 0.01, 1.0, 0.1)}},
    "HistGradient Boosting": {"cls": HistGradientBoostingClassifier, "fixed": {"random_state": 42},
                              "params": {"learning_rate": ("float", 0.01, 1.0, 0.1), "max_iter": ("int", 20, 500, 100)}},
    "AdaBoost": {"cls": AdaBoostClassifier, "fixed": {"random_state": 42},
                 "params": {"n_estimators": ("int", 10, 500, 50)}},
    "Naive Bayes": {"cls": GaussianNB, "fixed": {}, "params": {}},
    "Neural Network (MLP)": {"cls": MLPClassifier, "fixed": _MLP_FIXED, "params": _MLP_PARAMS},
}

REGRESSORS = {
    "Linear Regression": {"cls": LinearRegression, "fixed": {}, "params": {}},
    "Ridge": {"cls": Ridge, "fixed": {}, "params": {"alpha": ("float", 0.01, 50.0, 1.0)}},
    "Lasso": {"cls": Lasso, "fixed": {"max_iter": 5000}, "params": {"alpha": ("float", 0.001, 10.0, 0.1)}},
    "ElasticNet": {"cls": ElasticNet, "fixed": {"max_iter": 5000},
                   "params": {"alpha": ("float", 0.001, 10.0, 0.1), "l1_ratio": ("float", 0.0, 1.0, 0.5)}},
    "KNN": {"cls": KNeighborsRegressor, "fixed": {}, "params": {"n_neighbors": ("int", 1, 30, 5)}},
    "SVR": {"cls": SVR, "fixed": {}, "scale_y": True, "params": {"C": ("float", 0.01, 20.0, 1.0)}},
    "Decision Tree": {"cls": DecisionTreeRegressor, "fixed": {"random_state": 42},
                      "params": {"max_depth": ("int", 1, 30, 5)}},
    "Random Forest": {"cls": RandomForestRegressor, "fixed": {"random_state": 42, "n_jobs": -1},
                      "params": {"n_estimators": ("int", 10, 500, 100), "max_depth": ("int", 1, 30, 10)}},
    "Extra Trees": {"cls": ExtraTreesRegressor, "fixed": {"random_state": 42, "n_jobs": -1},
                    "params": {"n_estimators": ("int", 10, 500, 100), "max_depth": ("int", 1, 30, 10)}},
    "Gradient Boosting": {"cls": GradientBoostingRegressor, "fixed": {"random_state": 42},
                          "params": {"n_estimators": ("int", 10, 500, 100), "learning_rate": ("float", 0.01, 1.0, 0.1)}},
    "HistGradient Boosting": {"cls": HistGradientBoostingRegressor, "fixed": {"random_state": 42},
                              "params": {"learning_rate": ("float", 0.01, 1.0, 0.1), "max_iter": ("int", 20, 500, 100)}},
    "AdaBoost": {"cls": AdaBoostRegressor, "fixed": {"random_state": 42},
                 "params": {"n_estimators": ("int", 10, 500, 50)}},
    "Neural Network (MLP)": {"cls": MLPRegressor, "fixed": _MLP_FIXED, "scale_y": True, "params": _MLP_PARAMS},
}

# ---- Unsupervised ----
CLUSTERERS = {
    "KMeans": {"cls": KMeans, "fixed": {"random_state": 42, "n_init": 10},
               "params": {"n_clusters": ("int", 2, 20, 3)}},
    "MiniBatch KMeans": {"cls": MiniBatchKMeans, "fixed": {"random_state": 42, "n_init": 3},
                         "params": {"n_clusters": ("int", 2, 20, 3)}},
    "Agglomerative": {"cls": AgglomerativeClustering, "fixed": {},
                      "params": {"n_clusters": ("int", 2, 20, 3),
                                 "linkage": ("choice", ["ward", "complete", "average", "single"], "ward")}},
    "DBSCAN": {"cls": DBSCAN, "fixed": {},
               "params": {"eps": ("float", 0.1, 20.0, 2.0), "min_samples": ("int", 2, 50, 5)}},
    "Gaussian Mixture": {"cls": GaussianMixture, "fixed": {"random_state": 42},
                         "params": {"n_components": ("int", 2, 20, 3)}},
    "Birch": {"cls": Birch, "fixed": {}, "params": {"n_clusters": ("int", 2, 20, 3),
                                                     "threshold": ("float", 0.05, 5.0, 0.5)}},
}

ANOMALY = {
    "Isolation Forest": {"cls": IsolationForest, "fixed": {"random_state": 42},
                         "params": {"contamination": ("float", 0.01, 0.5, 0.05),
                                    "n_estimators": ("int", 50, 500, 100)}},
    "Local Outlier Factor": {"cls": LocalOutlierFactor, "fixed": {},
                             "params": {"contamination": ("float", 0.01, 0.5, 0.05),
                                        "n_neighbors": ("int", 5, 100, 20)}},
    "One-Class SVM": {"cls": OneClassSVM, "fixed": {}, "params": {"nu": ("float", 0.01, 0.5, 0.05)}},
    "Elliptic Envelope": {"cls": EllipticEnvelope, "fixed": {"random_state": 42},
                          "params": {"contamination": ("float", 0.01, 0.5, 0.05)}},
}

DIMRED = {
    "PCA": {"cls": PCA, "fixed": {"random_state": 42}, "params": {"n_components": ("int", 2, 10, 2)}},
    "Truncated SVD": {"cls": TruncatedSVD, "fixed": {"random_state": 42},
                      "params": {"n_components": ("int", 2, 10, 2)}},
    "t-SNE": {"cls": TSNE, "fixed": {"random_state": 42, "init": "pca"},
              "params": {"perplexity": ("float", 5.0, 50.0, 30.0)}},
}

try:  # XGBoost اختياري
    from xgboost import XGBClassifier, XGBRegressor

    _xgb = {"n_estimators": ("int", 10, 500, 100), "learning_rate": ("float", 0.01, 1.0, 0.1)}
    CLASSIFIERS["XGBoost"] = {"cls": XGBClassifier, "fixed": {"random_state": 42, "eval_metric": "logloss"}, "params": _xgb}
    REGRESSORS["XGBoost"] = {"cls": XGBRegressor, "fixed": {"random_state": 42}, "params": _xgb}
except Exception:
    pass

TASKS = {
    "classification": {"label": "تصنيف (Classification)", "reg": CLASSIFIERS, "supervised": True},
    "regression": {"label": "تنبؤ رقمي (Regression)", "reg": REGRESSORS, "supervised": True},
    "clustering": {"label": "تجميع (Clustering)", "reg": CLUSTERERS, "supervised": False},
    "anomaly": {"label": "كشف الشواذ (Anomaly)", "reg": ANOMALY, "supervised": False},
    "dimred": {"label": "تقليل الأبعاد (PCA / t-SNE)", "reg": DIMRED, "supervised": False},
}

# خوارزميات بطيئة/تقيلة على الداتا الكبيرة — بنحذّر المستخدم
SLOW_ALGOS = {"SVM", "SVR", "Gradient Boosting", "Neural Network (MLP)", "Agglomerative",
              "Local Outlier Factor", "One-Class SVM", "t-SNE", "Elliptic Envelope", "KNN"}
SLOW_ROWS = 20000


def get_registry(task: str) -> dict:
    return TASKS[task]["reg"]


def is_supervised(task: str) -> bool:
    return TASKS[task]["supervised"]


def parse_layers(txt) -> tuple:
    """'64,32' -> (64, 32)"""
    if isinstance(txt, (tuple, list)):
        return tuple(int(v) for v in txt)
    nums = [int(p) for p in str(txt).replace(" ", "").split(",") if p.strip().isdigit() and int(p) > 0]
    return tuple(nums) if nums else (64,)


def default_params(task: str, name: str) -> dict:
    out = {}
    for p, spec in get_registry(task)[name]["params"].items():
        out[p] = spec[1] if spec[0] == "layers" else (spec[2] if spec[0] == "choice" else spec[3])
    return out


def clean_params(task: str, name: str, user_params: dict | None) -> dict:
    """يحوّل قيم الواجهة لقيم sklearn (مثلاً layers نص -> tuple)."""
    spec = get_registry(task)[name]["params"]
    out = {}
    for k, v in (user_params or {}).items():
        out[k] = parse_layers(v) if spec.get(k, ("",))[0] == "layers" else v
    return out


def model_params(task: str, name: str, user_params: dict | None = None) -> dict:
    """fixed + defaults + قيم المستخدم (جاهزة تتبعت للـ constructor)."""
    spec = get_registry(task)[name]
    return {**spec["fixed"], **clean_params(task, name, {**default_params(task, name), **(user_params or {})})}


def supports_class_weight(cls) -> bool:
    try:
        return "class_weight" in cls().get_params()
    except Exception:
        return False


def build_model(task: str, name: str, user_params: dict | None = None, balanced: bool = False):
    spec = get_registry(task)[name]
    p = model_params(task, name, user_params)
    if balanced and task == "classification" and supports_class_weight(spec["cls"]):
        p["class_weight"] = "balanced"
    model = spec["cls"](**p)
    if spec.get("scale_y"):   # SVR / MLP بيتحسنوا جدًا لو الـ target اتعمله Scaling
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    return model


def import_line(cls) -> str:
    parts = cls.__module__.split(".")
    mod = ".".join(parts[:2]) if parts[0] == "sklearn" else parts[0]
    return f"from {mod} import {cls.__name__}"


# ==================== PREPROCESSING (scalers / encoders / Auto) ====================
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler,
)

try:  # Target Encoding (sklearn >= 1.3) — للأعمدة اللي فيها قيم كتير
    from sklearn.preprocessing import TargetEncoder
    HAS_TARGET_ENC = True
except Exception:
    TargetEncoder, HAS_TARGET_ENC = None, False

try:  # SMOTE اختياري
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except Exception:
    SMOTE = ImbPipeline = None
    HAS_IMB = False

IMPUTE_NUM = ["median", "mean", "most_frequent", "constant"]
IMPUTE_CAT = ["most_frequent", "constant"]
SCALERS = {"None": None, "Standard": StandardScaler, "MinMax": MinMaxScaler,
           "Robust": RobustScaler, "MaxAbs": MaxAbsScaler}
ENCODERS = ["onehot", "ordinal"] + (["target"] if HAS_TARGET_ENC else [])
HIGH_CARD = 10          # أكتر من كده = High Cardinality (مفيش One-Hot)
MISS_FLAG = 0.05        # لو الـ Missing أكتر من 5% بنضيف Missing-Indicator / فئة "missing"
KIND_OPTS = ["auto", "numeric", "categorical", "drop"]


def column_kind(s: pd.Series) -> str:
    if is_bool_dtype(s):
        return "categorical"
    return "numeric" if is_numeric_dtype(s) else "categorical"


def is_id_like(s: pd.Series) -> bool:
    """عمود قيمته مختلفة في كل صف (ID) — نصي أو أرقام صحيحة."""
    n = int(s.notna().sum())
    return n > 20 and s.nunique(dropna=True) == n and (column_kind(s) == "categorical" or is_integer_dtype(s))


def auto_excluded(df: pd.DataFrame, cols) -> list:
    """أعمدة بتتشال أوتوماتيك من الـ Features: ID أو قيمة واحدة."""
    return [c for c in cols if df[c].nunique(dropna=True) <= 1 or is_id_like(df[c])]


def _to_cat_str(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
            and float(v).is_integer():
        return str(int(v))
    return str(v)


def auto_config(s: pd.Series, supervised: bool = True) -> dict:
    miss = float(s.isna().mean())
    if column_kind(s) == "numeric":
        skew = s.dropna().skew() if s.notna().sum() > 2 else 0
        scale = "Robust" if abs(skew) > 1.5 else "Standard"
        return {"kind": "numeric", "impute": "median", "scale": scale, "encode": None, "indicator": miss > MISS_FLAG}
    n_unique = s.nunique(dropna=True)
    if n_unique <= HIGH_CARD:
        encode = "onehot"
    elif supervised and HAS_TARGET_ENC:
        encode = "target"           # High Cardinality: بدل One-Hot اللي بيفجّر الميموري
    else:
        encode = "ordinal"
    scale = "Standard" if encode == "ordinal" else "None"
    return {"kind": "categorical", "impute": "constant" if miss > MISS_FLAG else "most_frequent",
            "scale": scale, "encode": encode, "indicator": False}


def _imputer(cfg: dict):
    imp, ind = cfg.get("impute", "median"), bool(cfg.get("indicator", False))
    if cfg["kind"] == "numeric":
        if imp == "constant":
            return SimpleImputer(strategy="constant", fill_value=0, add_indicator=ind)
        return SimpleImputer(strategy=imp, add_indicator=ind)
    if imp == "constant":
        return SimpleImputer(strategy="constant", fill_value="missing")
    return SimpleImputer(strategy="most_frequent")


def _column_pipeline(cfg: dict) -> Pipeline:
    steps = [("impute", _imputer(cfg))]
    if cfg["kind"] == "categorical":
        enc = cfg.get("encode", "onehot")
        if enc == "onehot":
            steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif enc == "target" and HAS_TARGET_ENC:
            steps.append(("encode", TargetEncoder(random_state=42)))
        else:
            steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    scaler = SCALERS.get(cfg.get("scale", "None"))
    if scaler is not None:
        steps.append(("scale", scaler()))
    return Pipeline(steps)


def prepare_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """bool/كائنات مختلطة -> نص (والـ Missing بيفضل NaN) قبل الـ pipeline."""
    X = df[cols].copy()
    for c in cols:
        if column_kind(X[c]) == "categorical":
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))
    return X


def build_preprocessor(X: pd.DataFrame, config: dict) -> ColumnTransformer:
    transformers = [(f"col_{i}", _column_pipeline(config[c]), [c]) for i, c in enumerate(X.columns)]
    return ColumnTransformer(transformers, sparse_threshold=0)


def build_config(X: pd.DataFrame, auto: bool, manual: dict | None = None, supervised: bool = True) -> dict:
    config = {}
    for c in X.columns:
        base = auto_config(X[c], supervised)
        if not auto and manual and c in manual:
            base.update({k: v for k, v in manual[c].items() if v is not None or k == "encode"})
            if base["kind"] == "numeric":
                base["encode"] = None
            else:
                if base.get("encode") not in ENCODERS:
                    base["encode"] = "onehot"
                if base["encode"] == "target" and not supervised:
                    base["encode"] = "ordinal"
        config[c] = base
    return config


def encoded_target(df: pd.DataFrame, target: str, task: str) -> pd.Series:
    if task == "classification":
        return pd.Series(pd.factorize(df[target].astype(str))[0], index=df.index)
    return pd.to_numeric(df[target], errors="coerce")


# ==================== CLEANING + OUTLIERS ====================
OUTLIER_METHODS = {
    "none": "بدون معالجة",
    "iqr": "IQR (الأشهر — بيشتغل كويس مع أي توزيع)",
    "zscore": "Z-Score (للتوزيع الطبيعي)",
    "isolation": "Isolation Forest (متعدد الأعمدة)",
}
OUTLIER_ACTIONS = {
    "clip": "قصّ القيم للحد الأقصى/الأدنى (Winsorize) — مفيش صفوف بتتحذف",
    "median": "استبدال بالوسيط (Median)",
    "nan": "تحويلها لـ Missing (تتعوّض في التجهيز)",
    "remove": "حذف الصفوف",
}

DEFAULT_CLEAN = {
    "fix_types": True, "drop_dupes": True, "drop_cols": [], "max_missing_pct": 60,
    "outlier_method": "none", "outlier_action": "clip", "iqr_k": 1.5, "z_thr": 3.0,
    "contamination": 0.05, "outlier_cols": None,   # None = كل الأعمدة الرقمية
}


def numeric_cols(df: pd.DataFrame, exclude=()) -> list:
    return [c for c in df.columns if c not in exclude and column_kind(df[c]) == "numeric"]


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_mask(s: pd.Series, method: str, k: float = 1.5, z: float = 3.0) -> pd.Series:
    """True = قيمة شاذة."""
    s = pd.to_numeric(s, errors="coerce")
    if method == "zscore":
        sd = s.std()
        return pd.Series(False, index=s.index) if not sd or np.isnan(sd) else ((s - s.mean()).abs() / sd > z)
    lo, hi = iqr_bounds(s, k)
    return (s < lo) | (s > hi)


def count_outliers(df: pd.DataFrame, cols: list, method: str = "iqr", k: float = 1.5, z: float = 3.0) -> pd.DataFrame:
    rows = []
    for c in cols:
        m = outlier_mask(df[c], "iqr" if method not in ("iqr", "zscore") else method, k, z)
        rows.append({"Column": c, "Outliers": int(m.sum()), "Pct": round(100 * m.mean(), 2)})
    return pd.DataFrame(rows)


def clean_data(df: pd.DataFrame, cfg: dict, target: str | None = None):
    """يرجّع (df_clean, log). التنضيف بالترتيب: أنواع ← تكرار ← أعمدة ← Outliers.
    (ملحوظة: ده للعرض والتحليل. وقت تدريب الـ Supervised الـ Outliers بتتعالج بعد الـ split.)"""
    cfg = {**DEFAULT_CLEAN, **(cfg or {})}
    out, log = df.copy(), []

    if cfg["fix_types"]:
        for c in out.columns:
            if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
                conv = pd.to_numeric(out[c], errors="coerce")
                if out[c].notna().sum() and conv.notna().sum() / out[c].notna().sum() >= 0.9:
                    out[c] = conv
                    log.append(f"تحويل العمود «{c}» من نص لأرقام")

    if cfg["drop_dupes"]:
        n = int(out.duplicated().sum())
        if n:
            out = out.drop_duplicates().reset_index(drop=True)
            log.append(f"حذف {n} صف مكرر")

    drop = [c for c in cfg["drop_cols"] if c in out.columns and c != target]
    if drop:
        out = out.drop(columns=drop)
        log.append(f"حذف أعمدة يدويًا: {', '.join(map(str, drop))}")

    thr = cfg["max_missing_pct"]
    if thr < 100:
        miss = out.isna().mean() * 100
        auto_drop = [c for c in out.columns if miss[c] > thr and c != target]
        if auto_drop:
            out = out.drop(columns=auto_drop)
            log.append(f"حذف أعمدة Missing فيها أكتر من {thr}%: {', '.join(map(str, auto_drop))}")

    if target is not None and target in out.columns:
        n = int(out[target].isna().sum())
        if n:
            out = out[out[target].notna()].reset_index(drop=True)
            log.append(f"حذف {n} صف الـ Target فيه فاضي")

    method, action = cfg["outlier_method"], cfg["outlier_action"]
    if method != "none":
        pool = numeric_cols(out, exclude=[target] if target else [])
        cols = pool if cfg["outlier_cols"] is None else [c for c in cfg["outlier_cols"] if c in pool]
        if cols:
            if method == "isolation":
                data = out[cols].fillna(out[cols].median())
                flag = IsolationForest(contamination=cfg["contamination"], random_state=42).fit_predict(data) == -1
                n = int(flag.sum())
                out = out[~flag].reset_index(drop=True)
                log.append(f"Isolation Forest: حذف {n} صف شاذ")
            else:
                if action == "remove":
                    flag = pd.Series(False, index=out.index)
                    for c in cols:
                        flag |= outlier_mask(out[c], method, cfg["iqr_k"], cfg["z_thr"]).fillna(False)
                    n = int(flag.sum())
                    out = out[~flag].reset_index(drop=True)
                    log.append(f"{method.upper()}: حذف {n} صف فيه قيم شاذة")
                else:
                    total = 0
                    for c in cols:
                        m = outlier_mask(out[c], method, cfg["iqr_k"], cfg["z_thr"]).fillna(False)
                        k = int(m.sum())
                        if not k:
                            continue
                        total += k
                        s = pd.to_numeric(out[c], errors="coerce")
                        if action == "clip":
                            if method == "zscore":
                                lo, hi = s.mean() - cfg["z_thr"] * s.std(), s.mean() + cfg["z_thr"] * s.std()
                            else:
                                lo, hi = iqr_bounds(s, cfg["iqr_k"])
                            out[c] = s.clip(lo, hi)
                        elif action == "median":
                            out[c] = s.mask(m, s.median())
                        else:
                            out[c] = s.mask(m, np.nan)
                    verb = {"clip": "قصّ", "median": "استبدال بالوسيط", "nan": "تحويل لـ Missing"}[action]
                    log.append(f"{method.upper()}: {verb} لـ {total} قيمة شاذة في {len(cols)} عمود")
    return out, log


def _fit_bounds(s: pd.Series, method: str, k: float, z: float):
    """الحدود بتتحسب من الـ Train بس."""
    s = pd.to_numeric(s, errors="coerce")
    if method == "zscore":
        sd = s.std()
        if not sd or np.isnan(sd):
            return -np.inf, np.inf
        return s.mean() - z * sd, s.mean() + z * sd
    return iqr_bounds(s, k)


def apply_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict, target: str | None = None):
    """معالجة الـ Outliers من غير Data Leakage:
    الحدود/الوسيط/الموديل بيتحسبوا من الـ Train بس، وبعدين بتتطبق على الـ Train والـ Test.
    (حذف الصفوف بيتم من الـ Train بس — الـ Test بيفضل زي ما هو.)  يرجّع (train, test, log)."""
    cfg = {**DEFAULT_CLEAN, **(cfg or {})}
    method, action = cfg["outlier_method"], cfg["outlier_action"]
    tr, te, log = train_df.copy(), test_df.copy(), []
    if method == "none":
        return tr, te, log
    pool = numeric_cols(tr, exclude=[target] if target else [])
    cols = pool if cfg["outlier_cols"] is None else [c for c in cfg["outlier_cols"] if c in pool]
    if not cols:
        return tr, te, log

    if method == "isolation":
        med = tr[cols].median()
        iso = IsolationForest(contamination=cfg["contamination"], random_state=42).fit(tr[cols].fillna(med))
        flag = iso.predict(tr[cols].fillna(med)) == -1
        log.append(f"Isolation Forest: حذف {int(flag.sum())} صف شاذ من الـ Train بس")
        return tr[~flag], te, log

    bounds = {c: _fit_bounds(tr[c], method, cfg["iqr_k"], cfg["z_thr"]) for c in cols}
    if action == "remove":
        flag = pd.Series(False, index=tr.index)
        for c in cols:
            lo, hi = bounds[c]
            s_ = pd.to_numeric(tr[c], errors="coerce")
            flag |= ((s_ < lo) | (s_ > hi)).fillna(False)
        log.append(f"{method.upper()}: حذف {int(flag.sum())} صف شاذ من الـ Train بس")
        return tr[~flag], te, log

    total = 0
    for c in cols:
        lo, hi = bounds[c]
        med = pd.to_numeric(tr[c], errors="coerce").median()
        for part in (tr, te):
            s_ = pd.to_numeric(part[c], errors="coerce")
            m = ((s_ < lo) | (s_ > hi)).fillna(False)
            if part is tr:
                total += int(m.sum())
            if action == "clip":
                part[c] = s_.clip(lo, hi)
            elif action == "median":
                part[c] = s_.mask(m, med)
            else:
                part[c] = s_.mask(m, np.nan)
    verb = {"clip": "قصّ", "median": "استبدال بالوسيط", "nan": "تحويل لـ Missing"}[action]
    log.append(f"{method.upper()}: {verb} (الحدود من الـ Train بس) — {total} قيمة شاذة في الـ Train")
    return tr, te, log


# ==================== TRAINING (Supervised + Unsupervised) ====================
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

CLS_METRICS = ["Accuracy", "F1 (macro)", "Balanced Acc", "PR-AUC"]
SCORING = {"Accuracy": "accuracy", "F1 (macro)": "f1_macro", "Balanced Acc": "balanced_accuracy",
           "PR-AUC": "average_precision", "R2": "r2"}
IMBALANCE_MODES = {"none": "بدون معالجة",
                   "class_weight": "class_weight='balanced' (للموديلات اللي بتدعمه)",
                   "smote": "SMOTE (محتاج imbalanced-learn)"}
IMB_THRESHOLD = 0.2     # أصغر كلاس أقل من 20% = داتا غير متوازنة


def is_imbalanced(y) -> bool:
    vc = pd.Series(y).value_counts(normalize=True)
    return len(vc) > 1 and float(vc.min()) < IMB_THRESHOLD


def default_metric(y) -> str:
    return "F1 (macro)" if is_imbalanced(y) else "Accuracy"


def metric_options(y) -> list:
    return CLS_METRICS if pd.Series(y).nunique() == 2 else CLS_METRICS[:3]


def score_metric(metric: str, pipe, X, y, pos: int = 1) -> float:
    """pos = الكلاس الأقل (بيتحسب عليه PR-AUC)."""
    if metric == "PR-AUC":
        classes = list(pipe.classes_)
        if hasattr(pipe, "predict_proba"):
            s = pipe.predict_proba(X)[:, classes.index(pos)]
        else:
            d = np.asarray(pipe.decision_function(X))
            s = d if pos == classes[1] else -d
        return float(skm.average_precision_score(np.asarray(y) == pos, s))
    pred = pipe.predict(X)
    if metric == "F1 (macro)":
        return float(skm.f1_score(y, pred, average="macro"))
    if metric == "Balanced Acc":
        return float(skm.balanced_accuracy_score(y, pred))
    return float(skm.accuracy_score(y, pred))


def metric_baseline(metric: str, ytr) -> float:
    share = pd.Series(ytr).value_counts(normalize=True)
    if metric == "Accuracy":
        return float(share.max())
    if metric == "PR-AUC":
        return float(share.min())
    return 1.0 / max(len(share), 1)


def clean_feature_names(pre) -> list:
    out = []
    for n in pre.get_feature_names_out():
        out.append(n.split("__", 1)[1] if n.startswith("col_") and "__" in n else n)
    return out


def cap_rows(d: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """حد أقصى للصفوف عشان السيرفرات المجانية متقعش."""
    return d if len(d) <= max_rows else d.sample(max_rows, random_state=42).sort_index()


FIT_GAP = 0.15        # فرق Train - Test أكبر منه = احتمال Overfitting
FIT_FLAGS = {"over": "⚠️ Overfitting محتمل", "under": "⚠️ Underfitting محتمل", "ok": "✅ متوازن"}


def fit_diagnosis(task: str, train: float, test: float, baseline: float = 0.0) -> str:
    """بيقارن أداء الـ Train بالـ Test. baseline = أداء الموديل الغبي (للتصنيف)."""
    if train - test > FIT_GAP:
        return FIT_FLAGS["over"]
    weak = train < baseline + 0.05 if task == "classification" else train < 0.3
    return FIT_FLAGS["under"] if weak else FIT_FLAGS["ok"]


def train_supervised(df, target, features, task, chosen, user_params, config, test_size, do_cv, progress=None,
                     outlier_cfg=None, metric=None, imbalance="none"):
    """لو outlier_cfg فيه method غير none: لازم df يكون قبل معالجة الـ Outliers،
    والمعالجة بتتم بعد الـ split وبتتحسب من الـ Train بس (من غير Data Leakage).
    كل التجهيز (Imputer/Scaler/Encoder/TargetEncoder/SMOTE) جوه Pipeline بيتعمله fit على الـ Train بس."""
    df = df.copy()
    y = df[target]
    is_cls = task == "classification"
    if is_cls:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    else:
        le = None
        y = pd.to_numeric(y, errors="coerce")
        ok = y.notna()
        df, y = df[ok], y[ok]

    strat = y if (is_cls and y.value_counts().min() >= 2) else None
    idx_tr, idx_te = train_test_split(df.index, test_size=test_size, random_state=42, stratify=strat)
    df_tr, df_te = df.loc[idx_tr], df.loc[idx_te]
    if outlier_cfg and outlier_cfg.get("outlier_method", "none") != "none":
        df_tr, df_te, _ = apply_outliers(df_tr, df_te, outlier_cfg, target)
    Xtr, Xte = prepare_features(df_tr, features), prepare_features(df_te, features)
    ytr, yte = y.loc[df_tr.index], y.loc[df_te.index]

    key = "R2"
    pos, binary = 1, False
    if is_cls:
        binary = ytr.nunique() == 2
        key = metric if metric in CLS_METRICS else "Accuracy"
        if key == "PR-AUC" and not binary:
            key = "F1 (macro)"
        pos = int(ytr.value_counts().idxmin())
    balanced = is_cls and imbalance == "class_weight"
    smote_k = int(ytr.value_counts().min()) - 1 if is_cls else 0
    use_smote = is_cls and imbalance == "smote" and HAS_IMB and smote_k >= 1
    smote_k = min(5, smote_k)
    scoring = SCORING[key]

    results, fitted = [], {}
    for i, name in enumerate(chosen):
        try:
            prep = build_preprocessor(Xtr, config)
            model = build_model(task, name, user_params.get(name), balanced)
            if use_smote:
                pipe = ImbPipeline([("prep", prep), ("smote", SMOTE(k_neighbors=smote_k, random_state=42)),
                                    ("model", model)])
            else:
                pipe = Pipeline([("prep", prep), ("model", model)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            if is_cls:
                row = {"Algorithm": name,
                       "Accuracy": float(skm.accuracy_score(yte, pred)),
                       "F1 (macro)": float(skm.f1_score(yte, pred, average="macro")),
                       "Balanced Acc": float(skm.balanced_accuracy_score(yte, pred))}
                if binary:
                    try:
                        row["PR-AUC"] = score_metric("PR-AUC", pipe, Xte, yte, pos)
                    except Exception:
                        row["PR-AUC"] = np.nan
                if balanced:
                    row["class_weight"] = "✓" if supports_class_weight(get_registry(task)[name]["cls"]) else "—"
                te_score = row[key]
                tr_score = score_metric(key, pipe, Xtr, ytr, pos)
                baseline = metric_baseline(key, ytr)
            else:
                ptr = pipe.predict(Xtr)
                row = {"Algorithm": name, "R2": float(skm.r2_score(yte, pred)),
                       "RMSE": float(np.sqrt(skm.mean_squared_error(yte, pred))),
                       "MAE": float(skm.mean_absolute_error(yte, pred))}
                tr_score, te_score, baseline = float(skm.r2_score(ytr, ptr)), row["R2"], 0.0
            row["Train score"] = tr_score
            row["Gap"] = tr_score - te_score
            row["Fit"] = fit_diagnosis(task, tr_score, te_score, baseline)
            if do_cv:
                try:
                    cv = max(2, min(5, int(ytr.value_counts().min()))) if is_cls else 5
                    row["CV mean"] = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring).mean()
                except Exception:
                    row["CV mean"] = np.nan
            results.append(row)
            fitted[name] = pipe
        except Exception as e:
            results.append({"Algorithm": name, "Error": str(e)[:120]})
        if progress:
            progress((i + 1) / len(chosen))
    return {"table": pd.DataFrame(results), "fitted": fitted, "task": task, "yte": yte, "Xte": Xte,
            "le": le, "features": features, "config": config, "target": target, "key": key,
            "scoring": scoring, "pos": pos, "imbalanced": bool(is_cls and is_imbalanced(ytr)),
            "imbalance": "smote" if use_smote else ("class_weight" if balanced else "none"),
            "n_train": len(Xtr), "n_test": len(Xte)}


def cluster_metrics(Z, labels) -> dict:
    mask = labels != -1
    k = len(set(labels[mask]))
    row = {"Clusters": k, "Noise": int((~mask).sum()), "Silhouette": np.nan,
           "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    if k >= 2 and mask.sum() > k:
        Zm, lm = Z[mask], labels[mask]
        row["Silhouette"] = skm.silhouette_score(Zm, lm, sample_size=min(5000, len(Zm)), random_state=42)
        row["Davies-Bouldin"] = skm.davies_bouldin_score(Zm, lm)
        row["Calinski-Harabasz"] = skm.calinski_harabasz_score(Zm, lm)
    return row


def train_unsupervised(df, features, task, chosen, user_params, config, progress=None):
    X = prepare_features(df, features)
    pre = build_preprocessor(X, config)
    Z = pre.fit_transform(X)
    proj = PCA(n_components=2, random_state=42).fit_transform(Z) if Z.shape[1] >= 2 else np.c_[Z[:, 0], np.zeros(len(Z))]
    rows, outputs = [], {}
    for i, name in enumerate(chosen):
        try:
            model = build_model(task, name, user_params.get(name))
            if task == "clustering":
                labels = np.asarray(model.fit_predict(Z))
                rows.append({"Algorithm": name, **cluster_metrics(Z, labels)})
                outputs[name] = labels
            elif task == "anomaly":
                labels = np.asarray(model.fit_predict(Z))
                n = int((labels == -1).sum())
                rows.append({"Algorithm": name, "Anomalies": n, "Pct": round(100 * n / len(Z), 2)})
                outputs[name] = labels
            else:
                idx = np.arange(len(Z))
                if name == "t-SNE" and len(Z) > 3000:
                    idx = np.sort(np.random.RandomState(42).choice(len(Z), 3000, replace=False))
                if name == "t-SNE":
                    emb = model.set_params(perplexity=min(model.perplexity, max(5.0, (len(idx) - 1) / 3))).fit_transform(Z[idx])
                else:
                    emb = model.fit_transform(Z[idx])
                row = {"Algorithm": name, "Components": emb.shape[1]}
                if hasattr(model, "explained_variance_ratio_"):
                    row["Explained variance"] = float(np.sum(model.explained_variance_ratio_))
                if hasattr(model, "kl_divergence_"):
                    row["KL divergence"] = float(model.kl_divergence_)
                rows.append(row)
                outputs[name] = {"emb": emb, "idx": idx}
        except Exception as e:
            rows.append({"Algorithm": name, "Error": str(e)[:120]})
        if progress:
            progress((i + 1) / len(chosen))
    return {"table": pd.DataFrame(rows), "outputs": outputs, "proj": proj, "task": task,
            "index": X.index, "features": features, "config": config, "feature_names": clean_feature_names(pre)}


def silhouette_by_k(df, features, config, kmax=10):
    X = prepare_features(df, features)
    Z = build_preprocessor(X, config).fit_transform(X)
    if len(Z) > 4000:
        Z = Z[np.random.RandomState(42).choice(len(Z), 4000, replace=False)]
    ks, sil, inertia = [], [], []
    for k in range(2, min(kmax, len(Z) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(Z)
        ks.append(k)
        sil.append(skm.silhouette_score(Z, km.labels_))
        inertia.append(km.inertia_)
    return ks, sil, inertia


# ==================== ANALYSIS (insights + plots + reports) ====================
PALETTE = ["#0E9F82", "#F2994A", "#3D6FD8", "#8E5CD9", "#D64550", "#12A5C0", "#7C8F8A", "#C25FB3"]
INK, MUTED, GRID = "#16302B", "#5B6F6A", "#DCE6E2"
SOFT = "#BFD0CA"


def _fig(w=7.0, h=3.4, rows=1, cols=1):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=110, facecolor="white")
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.ravel():
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    return fig, axes


def _fig1(w, h):
    fig, axes = _fig(w, h)
    return fig, axes[0, 0]


def _grid(n, per_row=3):
    return (n + per_row - 1) // per_row, min(n, per_row)


def plot_missing(df):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(25)
    fig, ax = _fig1(7, max(2.0, 0.3 * len(miss) + 1))
    if len(miss):
        ax.barh(miss.index[::-1].astype(str), miss.values[::-1], color=PALETTE[1])
        ax.set_xlabel("% missing", color=MUTED, fontsize=8)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", color=MUTED)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_hist_grid(df, cols):
    cols = list(cols)[:12]
    r, c = _grid(len(cols))
    fig, axes = _fig(3.2 * c, 2.3 * r, r, c)
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col].dropna(), bins=25, color=PALETTE[0], alpha=0.9)
        ax.set_title(str(col), fontsize=9, color=INK, loc="left")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_box_grid(df, cols):
    cols = list(cols)[:12]
    r, c = _grid(len(cols), 4)
    fig, axes = _fig(2.6 * c, 2.6 * r, r, c)
    for ax, col in zip(axes.ravel(), cols):
        ax.boxplot(df[col].dropna(), patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor="#CFEFE6", edgecolor=PALETTE[0]),
                   medianprops=dict(color=INK), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                   flierprops=dict(marker="o", markerfacecolor=PALETTE[4], markeredgecolor="none", markersize=3.5))
        ax.set_title(str(col), fontsize=9, color=INK, loc="left")
        ax.set_xticks([])
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_corr(df, cols):
    cols = list(cols)[:25]
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = _fig1(min(9, 1.0 + 0.55 * n), min(8, 0.8 + 0.5 * n))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.grid(False)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def plot_target(df, target, task):
    fig, ax = _fig1(6, 2.6)
    if task == "regression":
        ax.hist(pd.to_numeric(df[target], errors="coerce").dropna(), bins=30, color=PALETTE[2])
    else:
        vc = df[target].astype(str).value_counts().head(20)
        ax.bar(vc.index, vc.values, color=PALETTE[2])
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Target: {target}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_cat_counts(df, cols):
    cols = list(cols)[:9]
    r, c = _grid(len(cols))
    fig, axes = _fig(3.4 * c, 2.6 * r, r, c)
    for ax, col in zip(axes.ravel(), cols):
        vc = df[col].astype(str).value_counts().head(8)
        ax.barh(vc.index[::-1], vc.values[::-1], color=PALETTE[3])
        ax.set_title(str(col), fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion(cm, labels=None):
    n_rows, n_cols = cm.shape
    fig, ax = _fig1(5, 4)
    ax.grid(False)
    ax.imshow(cm, cmap="Greens")
    thr = cm.max() / 2 if cm.size else 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thr else INK)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    if labels is not None and len(labels) == n_cols:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax.set_ylabel("True", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def plot_actual_pred(y, pred):
    fig, ax = _fig1(5, 4)
    ax.scatter(y, pred, s=14, color=PALETTE[0], alpha=0.65, edgecolor="none")
    lo, hi = min(np.min(y), np.min(pred)), max(np.max(y), np.max(pred))
    ax.plot([lo, hi], [lo, hi], color=PALETTE[4], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Actual", color=MUTED, fontsize=8)
    ax.set_ylabel("Predicted", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def plot_importance(names, values, top=15):
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = _fig1(6, max(2.2, 0.28 * len(order) + 0.8))
    ax.barh([str(names[i]) for i in order], np.asarray(values)[order], color=PALETTE[0])
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_scatter2d(xy, labels=None, title="", anomaly=False):
    fig, ax = _fig1(6, 4.2)
    ax.grid(False)
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=PALETTE[0], alpha=0.7, edgecolor="none")
    elif anomaly:
        norm = labels != -1
        ax.scatter(xy[norm, 0], xy[norm, 1], s=12, color=SOFT, alpha=0.8, edgecolor="none", label="normal")
        ax.scatter(xy[~norm, 0], xy[~norm, 1], s=26, color=PALETTE[4], edgecolor="white", linewidth=0.4, label="anomaly")
        ax.legend(fontsize=8, frameon=False)
    else:
        for i, l in enumerate(sorted(set(labels))):
            m = labels == l
            ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.8, edgecolor="none",
                       color=SOFT if l == -1 else PALETTE[i % len(PALETTE)],
                       label="noise" if l == -1 else f"cluster {l}")
        if len(set(labels)) <= 12:
            ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_line(xs, ys, title, color=PALETTE[0], ylabel=""):
    fig, ax = _fig1(4.2, 2.8)
    ax.plot(xs, ys, marker="o", color=color)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("k", color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def data_insights(df: pd.DataFrame, target: str | None = None) -> list:
    """(level, text) — level: ok / info / warn"""
    out = []
    n, m = df.shape
    if n < 100:
        out.append(("warn", f"الداتا صغيرة ({n} صف) — النتائج ممكن تبقى غير مستقرة."))
    if n > SLOW_ROWS:
        out.append(("info", f"الداتا كبيرة ({n:,} صف) — فيه حد أقصى للصفوف في صفحة الخوارزميات عشان السرعة والرام."))
    dup = int(df.duplicated().sum())
    if dup:
        out.append(("warn", f"فيه {dup} صف مكرر ({100 * dup / n:.1f}%)."))
    miss = df.isna().mean() * 100
    for c in miss[miss > 20].sort_values(ascending=False).index[:5]:
        out.append(("warn", f"العمود «{c}» فيه {miss[c]:.0f}% قيم ناقصة."))
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        out.append(("warn", f"أعمدة بقيمة واحدة ملهاش فايدة: {', '.join(map(str, const[:5]))}"))
    for c in df.columns:
        s = df[c]
        if c in const or c == target:
            continue
        if is_id_like(s):
            out.append(("warn", f"«{c}» قيمته مختلفة في كل صف (غالبًا ID) — بيتشال أوتوماتيك من الـ Features."))
        elif column_kind(s) == "categorical" and s.nunique() > 50:
            out.append(("info", f"«{c}» فيه {s.nunique()} قيمة مختلفة (High Cardinality) — بيتعمله Target Encoding بدل One-Hot."))
        elif is_integer_dtype(s) and 2 < s.nunique() <= 6 and n > 50:
            out.append(("info", f"«{c}» أرقام صحيحة بس {s.nunique()} قيم — لو هي فئات (تقييم/كود) غيّر نوعها لـ categorical من صفحة التجهيز."))
    nums = numeric_cols(df)
    skewed = [c for c in nums if df[c].notna().sum() > 2 and abs(df[c].skew()) > 2]
    if skewed:
        out.append(("info", f"أعمدة التواءها عالي (Skewed): {', '.join(map(str, skewed[:6]))} — Robust Scaler مناسب لها."))
    if nums:
        oc = count_outliers(df, nums)
        oc = oc[oc.Outliers > 0].sort_values("Outliers", ascending=False)
        if len(oc):
            top = ", ".join(f"{r.Column} ({r.Outliers})" for r in oc.head(4).itertuples())
            out.append(("warn", f"Outliers (IQR) في {len(oc)} عمود — أعلاها: {top}. اعمل معالجة من صفحة التنضيف."))
    if len(nums) > 1:
        corr = df[nums[:60]].corr().abs()
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
                 if corr.loc[a, b] > 0.9]
        for a, b, v in sorted(pairs, key=lambda t: -t[2])[:3]:
            out.append(("info", f"«{a}» و«{b}» مرتبطين جدًا (r={v:.2f}) — ممكن تحذف واحد منهم."))
    if target and target in df.columns:
        t = df[target]
        if column_kind(t) == "categorical" or t.nunique() <= 15:
            share = t.value_counts(normalize=True)
            if len(share) > 1 and share.min() < IMB_THRESHOLD:
                out.append(("warn", f"الـ Target غير متوازن: «{share.idxmin()}» ممثل بـ {100 * share.min():.1f}% بس — "
                                    "الـ Accuracy هتبقى مضللة، استخدم F1 / PR-AUC (هتتختار تلقائيًا) و class_weight أو SMOTE."))
    if not out:
        out.append(("ok", "الداتا شكلها نضيف — مفيش مشاكل واضحة."))
    return out


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Type": df.dtypes.astype(str), "Kind": [column_kind(df[c]) for c in df.columns],
        "Missing": df.isna().sum(), "Missing %": (df.isna().mean() * 100).round(2),
        "Unique": df.nunique(),
    })


def build_report_xlsx(df: pd.DataFrame, target: str | None = None):
    """شيت Excel متعدد الصفحات. بيرجّع None لو مفيش openpyxl/xlsxwriter."""
    nums = numeric_cols(df)
    sheets = {
        "Summary": pd.DataFrame({"Metric": ["Rows", "Columns", "Duplicates", "Missing cells", "Numeric cols",
                                            "Categorical cols"],
                                 "Value": [len(df), df.shape[1], int(df.duplicated().sum()), int(df.isna().sum().sum()),
                                           len(nums), df.shape[1] - len(nums)]}),
        "Columns": column_summary(df).reset_index(names="Column"),
        "Statistics": df.describe(include="all").T.reset_index(names="Column"),
        "Outliers": count_outliers(df, nums) if nums else pd.DataFrame({"Column": []}),
        "Correlation": df[nums].corr().round(3).reset_index(names="Column") if len(nums) > 1 else pd.DataFrame(),
        "Insights": pd.DataFrame(data_insights(df, target), columns=["Level", "Insight"]),
        "Preview": df.head(200),
    }
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine=engine) as xw:
                for name, sdf in sheets.items():
                    sdf.to_excel(xw, sheet_name=name, index=False)
                    ws = xw.sheets[name]
                    for i, col in enumerate(sdf.columns):
                        width = min(40, max(10, len(str(col)) + 2))
                        if engine == "openpyxl":
                            from openpyxl.utils import get_column_letter
                            ws.column_dimensions[get_column_letter(i + 1)].width = width
                        else:
                            ws.set_column(i, i, width)
            return buf.getvalue()
        except ImportError:
            continue
    return None


def build_report_html(df: pd.DataFrame, target: str | None = None, title: str = "Data Report") -> bytes:
    nums = numeric_cols(df)
    cats = [c for c in df.columns if c not in nums]
    imgs = [("Missing values", plot_missing(df))]
    if nums:
        imgs.append(("Distributions", plot_hist_grid(df, nums)))
        imgs.append(("Boxplots (Outliers)", plot_box_grid(df, nums)))
    if len(nums) > 1:
        imgs.append(("Correlation", plot_corr(df, nums)))
    small = [c for c in cats if df[c].nunique() <= 30]
    if small:
        imgs.append(("Categorical columns", plot_cat_counts(df, small)))
    ins = "".join(f"<li class='{l}'>{t}</li>" for l, t in data_insights(df, target))
    figs = "".join(f"<h2>{t}</h2><img src='data:image/png;base64,{fig_to_b64(f)}'/>" for t, f in imgs)
    css = ("body{font-family:'IBM Plex Sans Arabic',Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;color:#16302B;"
           "padding:0 16px}h1{margin-bottom:4px}h2{margin-top:34px;border-bottom:1px solid #DCE6E2;padding-bottom:6px}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #DCE6E2;padding:4px 8px}"
           "th{background:#F4F7F5}img{max-width:100%}li.warn{color:#9A5B00}li.ok{color:#17734F}li{margin:5px 0}")
    html = (f"<html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>"
            f"<h1>{title}</h1><p>{df.shape[0]} rows × {df.shape[1]} columns — {int(df.isna().sum().sum())} missing cells"
            f" — {int(df.duplicated().sum())} duplicates</p><h2>Insights</h2><ul>{ins}</ul>"
            f"<h2>Columns</h2>{column_summary(df).to_html()}<h2>Statistics</h2>{df.describe().T.round(3).to_html()}"
            f"{figs}</body></html>")
    return html.encode("utf-8")


# ==================== CODE GENERATOR (clean → visualization → preprocessing → ML) ====================
def _fmt(v) -> str:
    return repr(v)


def _imputer_code(cfg: dict) -> str:
    imp, ind = cfg.get("impute", "median"), bool(cfg.get("indicator", False))
    if cfg["kind"] == "numeric":
        extra = ", add_indicator=True" if ind else ""
        if imp == "constant":
            return f'("impute", SimpleImputer(strategy="constant", fill_value=0{extra}))'
        return f'("impute", SimpleImputer(strategy="{imp}"{extra}))'
    if imp == "constant":
        return '("impute", SimpleImputer(strategy="constant", fill_value="missing"))'
    return '("impute", SimpleImputer(strategy="most_frequent"))'


def _pipe_code(cfg: dict) -> str:
    steps = [_imputer_code(cfg)]
    if cfg["kind"] == "categorical":
        enc = cfg.get("encode", "onehot")
        steps.append({"onehot": '("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))',
                      "target": '("encode", TargetEncoder(random_state=42))'}.get(
            enc, '("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))'))
    sc = cfg.get("scale", "None")
    if sc != "None":
        steps.append(f'("scale", {SCALERS[sc].__name__}())')
    return "Pipeline([" + ", ".join(steps) + "])"


def _model_expr(task: str, name: str, user_params: dict | None, balanced: bool = False) -> str:
    spec = get_registry(task)[name]
    params = model_params(task, name, user_params)
    if balanced and task == "classification" and supports_class_weight(spec["cls"]):
        params["class_weight"] = "balanced"
    expr = f"{spec['cls'].__name__}(" + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items()) + ")"
    if spec.get("scale_y"):
        expr = f"TransformedTargetRegressor(regressor={expr}, transformer=StandardScaler())"
    return expr


def _clean_code(cfg: dict, target: str | None, with_outliers: bool = True) -> str:
    cfg = {**DEFAULT_CLEAN, **cfg}
    L = [f"TARGET = {_fmt(target)}", ""]
    if cfg["fix_types"]:
        L += ["# تحويل الأعمدة النصية اللي أغلبها أرقام",
              "for c in df.columns:",
              "    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):",
              '        conv = pd.to_numeric(df[c], errors="coerce")',
              "        if df[c].notna().sum() and conv.notna().sum() / df[c].notna().sum() >= 0.9:",
              "            df[c] = conv", ""]
    if cfg["drop_dupes"]:
        L += ["df = df.drop_duplicates().reset_index(drop=True)", ""]
    if cfg["drop_cols"]:
        L += [f"df = df.drop(columns=[c for c in {_fmt(list(cfg['drop_cols']))} if c in df.columns and c != TARGET])", ""]
    if cfg["max_missing_pct"] < 100:
        L += [f"# حذف الأعمدة اللي الـ Missing فيها أكتر من {cfg['max_missing_pct']}%",
              "miss_pct = df.isna().mean() * 100",
              f"df = df.drop(columns=[c for c in df.columns if miss_pct[c] > {cfg['max_missing_pct']} and c != TARGET])", ""]
    if target is not None:
        L += ["# حذف الصفوف اللي الـ Target فيها فاضي",
              "df = df[df[TARGET].notna()].reset_index(drop=True)", ""]
    m, a = cfg["outlier_method"], cfg["outlier_action"]
    if m != "none" and with_outliers:
        if cfg["outlier_cols"] is None:
            L.append('OUTLIER_COLS = [c for c in df.select_dtypes("number").columns if c != TARGET]')
        else:
            L.append(f"OUTLIER_COLS = [c for c in {_fmt(list(cfg['outlier_cols']))} if c in df.columns and c != TARGET]")
        if m == "isolation":
            L += ["# Isolation Forest: حذف الصفوف الشاذة",
                  "from sklearn.ensemble import IsolationForest",
                  "if OUTLIER_COLS:",
                  "    data = df[OUTLIER_COLS].fillna(df[OUTLIER_COLS].median())",
                  f"    flag = IsolationForest(contamination={cfg['contamination']}, random_state=42).fit_predict(data) == -1",
                  "    df = df[~flag].reset_index(drop=True)"]
        else:
            L += ["", "def bounds(s):"]
            if m == "iqr":
                L += ["    q1, q3 = s.quantile(0.25), s.quantile(0.75)",
                      "    iqr = q3 - q1",
                      f"    return q1 - {cfg['iqr_k']} * iqr, q3 + {cfg['iqr_k']} * iqr"]
            else:
                L += [f"    return s.mean() - {cfg['z_thr']} * s.std(), s.mean() + {cfg['z_thr']} * s.std()"]
            L.append("")
            if a == "remove":
                L += ["flag = pd.Series(False, index=df.index)",
                      "for c in OUTLIER_COLS:",
                      "    lo, hi = bounds(df[c])",
                      "    flag |= (df[c] < lo) | (df[c] > hi)",
                      "df = df[~flag].reset_index(drop=True)"]
            else:
                L += ["for c in OUTLIER_COLS:",
                      "    s = pd.to_numeric(df[c], errors='coerce')",
                      "    lo, hi = bounds(s)",
                      "    mask = (s < lo) | (s > hi)"]
                L.append({"clip": "    df[c] = s.clip(lo, hi)",
                          "median": "    df[c] = s.mask(mask, s.median())",
                          "nan": "    df[c] = s.mask(mask, np.nan)"}[a])
    L += ["", 'print("بعد التنضيف:", df.shape)']
    return "\n".join(L)


def _viz_code(task: str) -> str:
    L = ["import matplotlib.pyplot as plt", "",
         "num_cols = [c for c in df.select_dtypes('number').columns if c != TARGET]", "",
         "# 1) القيم الناقصة",
         "miss = (df.isna().mean() * 100).sort_values(ascending=False)",
         "miss = miss[miss > 0]",
         "if len(miss):",
         "    miss.plot.barh(figsize=(7, max(2, 0.3 * len(miss))), color='#F2994A', title='Missing %')",
         "    plt.show()",
         "else:",
         "    print('مفيش قيم ناقصة')", "",
         "# 2) التوزيعات (Histograms)",
         "if num_cols:",
         "    df[num_cols[:12]].hist(bins=25, figsize=(11, 7), color='#0E9F82')",
         "    plt.tight_layout(); plt.show()", "",
         "# 3) Boxplots لاكتشاف الـ Outliers",
         "if num_cols:",
         "    fig, axes = plt.subplots(1, min(len(num_cols), 8), figsize=(2.2 * min(len(num_cols), 8), 3.2), squeeze=False)",
         "    for ax, c in zip(axes[0], num_cols[:8]):",
         "        ax.boxplot(df[c].dropna()); ax.set_title(c, fontsize=9); ax.set_xticks([])",
         "    plt.tight_layout(); plt.show()", "",
         "# 4) مصفوفة الارتباط",
         "if len(num_cols) > 1:",
         "    corr = df[num_cols[:25]].corr()",
         "    plt.figure(figsize=(8, 6)); plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1); plt.colorbar()",
         "    plt.xticks(range(len(corr)), corr.columns, rotation=60, ha='right', fontsize=7)",
         "    plt.yticks(range(len(corr)), corr.columns, fontsize=7)",
         "    plt.title('Correlation'); plt.tight_layout(); plt.show()"]
    if task in ("classification", "regression"):
        L += ["", "# 5) توزيع الـ Target",
              "plt.figure(figsize=(6, 3))"]
        L.append("df[TARGET].astype(str).value_counts().plot.bar(color='#3D6FD8')" if task == "classification"
                 else "df[TARGET].plot.hist(bins=30, color='#3D6FD8')")
        L.append("plt.title(f'Target: {TARGET}'); plt.tight_layout(); plt.show()")
    return "\n".join(L)


def _kinds_code(kinds: dict | None) -> str:
    """تغيير أنواع الأعمدة اللي المستخدم اختارها يدويًا."""
    if not kinds:
        return ""
    L = ["# أنواع أعمدة اتغيّرت يدويًا"]
    for c, k in kinds.items():
        if k == "numeric":
            L.append(f"df[{c!r}] = pd.to_numeric(df[{c!r}], errors='coerce')")
        else:
            L.append(f"df[{c!r}] = df[{c!r}].map(lambda v: v if pd.isna(v) else "
                     "(str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)))")
    return "\n".join(L) + "\n\n"


def _prep_code(features: list, config: dict) -> str:
    groups = {}
    for c in features:
        cfg = config[c]
        groups.setdefault((cfg["kind"], cfg["impute"], cfg["scale"], cfg.get("encode"), bool(cfg.get("indicator"))),
                          []).append(c)
    L = [f"FEATURES = {_fmt(list(features))}", "X = df[FEATURES].copy()", "",
         "# أي عمود bool أو نصي بيتحوّل لنص عشان الـ Encoder (والـ Missing بيفضل NaN)",
         "for c in X.columns:",
         "    if (not pd.api.types.is_numeric_dtype(X[c])) or pd.api.types.is_bool_dtype(X[c]):",
         "        X[c] = X[c].astype('object').where(X[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))", "",
         "# كل خطوات التجهيز جوه الـ Pipeline => بيتعملها fit على الـ Train بس (من غير Data Leakage)",
         "preprocessor = ColumnTransformer(["]
    for i, ((kind, imp, sc, enc, ind), cols) in enumerate(groups.items()):
        cfg = {"kind": kind, "impute": imp, "scale": sc, "encode": enc, "indicator": ind}
        prefix = "num" if kind == "numeric" else "cat"
        L.append(f'    ("{prefix}_{i}", {_pipe_code(cfg)}, {_fmt(cols)}),')
    L += ["], sparse_threshold=0)"]
    return "\n".join(L)


def _imports(task, cfg_prep, chosen, features, config, clean) -> str:
    L = ["import warnings", "import numpy as np", "import pandas as pd",
         "from sklearn.base import clone", "from sklearn.compose import ColumnTransformer",
         "from sklearn.impute import SimpleImputer", "from sklearn.pipeline import Pipeline",
         "from sklearn import metrics as skm"]
    scalers = {config[c]["scale"] for c in features if config[c]["scale"] != "None"}
    enc = {config[c].get("encode") for c in features if config[c]["kind"] == "categorical"}
    pre = sorted([SCALERS[s].__name__ for s in scalers] +
                 (["OneHotEncoder"] if "onehot" in enc else []) + (["OrdinalEncoder"] if "ordinal" in enc else []) +
                 (["TargetEncoder"] if "target" in enc else []))
    reg = get_registry(task)
    needs_tt = any(reg[n].get("scale_y") for n in chosen)
    if needs_tt and "StandardScaler" not in pre:
        pre.append("StandardScaler")
    if task == "classification":
        pre.append("LabelEncoder")
    if pre:
        L.append("from sklearn.preprocessing import " + ", ".join(sorted(set(pre))))
    if needs_tt:
        L.append("from sklearn.compose import TransformedTargetRegressor")
    if task in ("classification", "regression"):
        L.append("from sklearn.model_selection import cross_val_score, train_test_split")
    elif task in ("clustering", "anomaly"):
        L.append("from sklearn.decomposition import PCA")
    L += sorted({import_line(reg[n]["cls"]) for n in chosen})
    L += ["", "warnings.filterwarnings('ignore')"]
    return "\n".join(L)


def _outlier_after_split_code(cfg: dict) -> list:
    """Outliers بعد الـ split — الحدود من الـ Train بس (من غير Data Leakage)."""
    cfg = {**DEFAULT_CLEAN, **cfg}
    m, a = cfg["outlier_method"], cfg["outlier_action"]
    L = ["", "# ---- Outliers: الحدود بتتحسب من الـ Train بس (عشان مفيش Data Leakage) ----",
         "X_train, X_test = X_train.copy(), X_test.copy()"]
    if cfg["outlier_cols"] is None:
        L.append('OUTLIER_COLS = list(X_train.select_dtypes("number").columns)')
    else:
        L.append(f"OUTLIER_COLS = [c for c in {_fmt(list(cfg['outlier_cols']))} "
                 "if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]")
    if m == "isolation":
        L += ["from sklearn.ensemble import IsolationForest",
              "if OUTLIER_COLS:",
              "    med = X_train[OUTLIER_COLS].median()",
              f"    iso = IsolationForest(contamination={cfg['contamination']}, random_state=42)"
              ".fit(X_train[OUTLIER_COLS].fillna(med))",
              "    keep = iso.predict(X_train[OUTLIER_COLS].fillna(med)) != -1",
              "    X_train, y_train = X_train[keep], y_train[keep]"]
        return L
    L += ["", "def bounds(s):"]
    if m == "iqr":
        L += ["    q1, q3 = s.quantile(0.25), s.quantile(0.75)", "    iqr = q3 - q1",
              f"    return q1 - {cfg['iqr_k']} * iqr, q3 + {cfg['iqr_k']} * iqr"]
    else:
        L += [f"    return s.mean() - {cfg['z_thr']} * s.std(), s.mean() + {cfg['z_thr']} * s.std()"]
    L.append("")
    if a == "remove":
        L += ["flag = pd.Series(False, index=X_train.index)",
              "for c in OUTLIER_COLS:",
              "    lo, hi = bounds(X_train[c])",
              "    flag |= (X_train[c] < lo) | (X_train[c] > hi)",
              "X_train, y_train = X_train[~flag], y_train[~flag]"]
    else:
        L += ["for c in OUTLIER_COLS:",
              "    lo, hi = bounds(X_train[c])",
              "    med = X_train[c].median()"]
        if a == "clip":
            L += ["    X_train[c] = X_train[c].clip(lo, hi)", "    X_test[c] = X_test[c].clip(lo, hi)"]
        else:
            fill = "med" if a == "median" else "np.nan"
            L += [f"    X_train[c] = X_train[c].mask((X_train[c] < lo) | (X_train[c] > hi), {fill})",
                  f"    X_test[c] = X_test[c].mask((X_test[c] < lo) | (X_test[c] > hi), {fill})"]
    return L


_SCORE_BODY = {
    "Accuracy": ["    return skm.accuracy_score(y_, pipe.predict(X_))"],
    "F1 (macro)": ["    return skm.f1_score(y_, pipe.predict(X_), average='macro')"],
    "Balanced Acc": ["    return skm.balanced_accuracy_score(y_, pipe.predict(X_))"],
    "PR-AUC": ["    if hasattr(pipe, 'predict_proba'):",
               "        s = pipe.predict_proba(X_)[:, list(pipe.classes_).index(POS)]",
               "    else:",
               "        s = pipe.decision_function(X_) * (1 if POS == 1 else -1)",
               "    return skm.average_precision_score(np.asarray(y_) == POS, s)"],
}


def _unsup_code(task, chosen) -> str:
    models = "MODELS = {\n" + "\n".join(f"    {_fmt(n)}: {_model_expr(task, n, None)}," for n in chosen) + "\n}"
    return _unsup_code_body(task, models)


def _unsup_code_body(task, models) -> str:
    L = ["Z = preprocessor.fit_transform(X)", models, "", "import matplotlib.pyplot as plt"]
    if task == "dimred":
        L += ["rows = []", "for name, model in MODELS.items():",
              "    idx = np.arange(len(Z))",
              "    if name == 't-SNE' and len(Z) > 3000:",
              "        idx = np.sort(np.random.RandomState(42).choice(len(Z), 3000, replace=False))",
              "    emb = model.fit_transform(Z[idx])",
              "    row = {'Algorithm': name, 'Components': emb.shape[1]}",
              "    if hasattr(model, 'explained_variance_ratio_'):",
              "        row['Explained variance'] = float(np.sum(model.explained_variance_ratio_))",
              "    rows.append(row)",
              "    plt.figure(figsize=(5, 4)); plt.scatter(emb[:, 0], emb[:, 1], s=12, alpha=.7)",
              "    plt.title(name); plt.show()",
              "print(pd.DataFrame(rows).to_string())"]
        return "\n".join(L)
    L += ["proj = PCA(n_components=2, random_state=42).fit_transform(Z) if Z.shape[1] >= 2 else np.c_[Z[:, 0], np.zeros(len(Z))]",
          "rows, outputs = [], {}", "for name, model in MODELS.items():",
          "    labels = np.asarray(model.fit_predict(Z))", "    outputs[name] = labels"]
    if task == "clustering":
        L += ["    mask = labels != -1", "    k = len(set(labels[mask]))",
              "    row = {'Algorithm': name, 'Clusters': k, 'Noise': int((~mask).sum())}",
              "    if k >= 2 and mask.sum() > k:",
              "        row['Silhouette'] = skm.silhouette_score(Z[mask], labels[mask], sample_size=min(5000, int(mask.sum())), random_state=42)",
              "        row['Davies-Bouldin'] = skm.davies_bouldin_score(Z[mask], labels[mask])",
              "        row['Calinski-Harabasz'] = skm.calinski_harabasz_score(Z[mask], labels[mask])",
              "    rows.append(row)",
              "    plt.figure(figsize=(5, 4)); plt.scatter(proj[:, 0], proj[:, 1], c=labels, s=12, cmap='tab10')",
              "    plt.title(f'{name} (PCA 2D)'); plt.show()", "",
              "results = pd.DataFrame(rows)", "print(results.round(4).to_string())", "",
              "# إضافة رقم الـ Cluster لأفضل موديل (أعلى Silhouette)",
              "best_name = results.sort_values('Silhouette', ascending=False).iloc[0]['Algorithm']",
              "df['cluster'] = outputs[best_name]",
              "print(df.groupby('cluster').mean(numeric_only=True).round(2))"]
    else:
        L += ["    n = int((labels == -1).sum())", "    rows.append({'Algorithm': name, 'Anomalies': n, 'Pct': round(100 * n / len(Z), 2)})",
              "    bad = labels == -1",
              "    plt.figure(figsize=(5, 4))",
              "    plt.scatter(proj[~bad, 0], proj[~bad, 1], s=12, c='#BFD0CA')",
              "    plt.scatter(proj[bad, 0], proj[bad, 1], s=26, c='#D64550')",
              "    plt.title(f'{name} (PCA 2D)'); plt.show()", "",
              "print(pd.DataFrame(rows).to_string())", "",
              "# إضافة علامة الشاذ لأول موديل",
              "df['is_anomaly'] = outputs[list(MODELS)[0]] == -1"]
    return "\n".join(L)


def _model_code(task, chosen, params, test_size, do_cv, outlier_cfg=None, metric=None, imbalance="none") -> str:
    if task not in ("classification", "regression"):
        return _unsup_code(task, chosen)
    is_cls = task == "classification"
    balanced = is_cls and imbalance == "class_weight"
    smote = is_cls and imbalance == "smote"
    key = (metric if metric in CLS_METRICS else "Accuracy") if is_cls else "R2"
    models = "MODELS = {\n" + "\n".join(
        f"    {_fmt(n)}: {_model_expr(task, n, params.get(n), balanced)}," for n in chosen) + "\n}"
    L = ["y = df[TARGET]"]
    if is_cls:
        L += ["le = LabelEncoder()", "y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)"]
    else:
        L += ["y = pd.to_numeric(y, errors='coerce')", "ok = y.notna()", "X, y = X[ok], y[ok]"]
    strat = "y if y.value_counts().min() >= 2 else None" if is_cls else "None"
    L += ["", "# الـ split الأول — وبعدها أي حاجة بتتعلّم من الداتا (Imputer/Scaler/Encoder/SMOTE) بتتعلّم من الـ Train بس",
          f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, "
          f"random_state=42, stratify={strat})"]
    if outlier_cfg and outlier_cfg.get("outlier_method", "none") != "none":
        L += _outlier_after_split_code(outlier_cfg)
    if is_cls:
        if key == "PR-AUC":
            L += ["", "POS = int(y_train.value_counts().idxmin())  # الكلاس الأقل"]
        L += ["", f"KEY = {key!r}  # المقياس الأساسي", "def score(pipe, X_, y_):"] + _SCORE_BODY[key]
    if smote:
        L += ["", "# SMOTE جوه الـ Pipeline: بيتطبق على الـ Train بس  (pip install imbalanced-learn)",
              "from imblearn.over_sampling import SMOTE",
              "from imblearn.pipeline import Pipeline as ImbPipeline",
              "K = max(1, min(5, int(y_train.value_counts().min()) - 1))"]
    L += ["", models, "", "rows, fitted = [], {}", "for name, model in MODELS.items():"]
    if smote:
        L.append("    pipe = ImbPipeline([('prep', clone(preprocessor)), "
                 "('smote', SMOTE(k_neighbors=K, random_state=42)), ('model', model)])")
    else:
        L.append("    pipe = Pipeline([('prep', clone(preprocessor)), ('model', model)])")
    L += ["    pipe.fit(X_train, y_train)", "    pred = pipe.predict(X_test)"]
    if is_cls:
        L += ["    row = {'Algorithm': name, 'Accuracy': skm.accuracy_score(y_test, pred),",
              "           'F1 (macro)': skm.f1_score(y_test, pred, average='macro'),",
              "           'Balanced Acc': skm.balanced_accuracy_score(y_test, pred)}",
              "    row[KEY] = score(pipe, X_test, y_test)",
              "    row['Train score'] = score(pipe, X_train, y_train)",
              "    row['Gap'] = row['Train score'] - row[KEY]  # لو أكبر من 0.15 = احتمال Overfitting"]
        cv = "max(2, min(5, int(y_train.value_counts().min())))"
    else:
        L += ["    row = {'Algorithm': name, 'R2': skm.r2_score(y_test, pred),",
              "           'RMSE': float(np.sqrt(skm.mean_squared_error(y_test, pred))),",
              "           'MAE': skm.mean_absolute_error(y_test, pred)}",
              "    row['Train score'] = skm.r2_score(y_train, pipe.predict(X_train))",
              "    row['Gap'] = row['Train score'] - row['R2']  # لو أكبر من 0.15 = احتمال Overfitting"]
        cv = "5"
    if do_cv:
        L.append(f"    row['CV mean'] = cross_val_score(pipe, X_train, y_train, cv={cv}, scoring='{SCORING[key]}').mean()")
    sort_key = key if is_cls else "R2"
    L += ["    rows.append(row); fitted[name] = pipe", "",
          f"results = pd.DataFrame(rows).sort_values({sort_key!r}, ascending=False).reset_index(drop=True)",
          "print(results.round(4).to_string())", "", "best_name = results.loc[0, 'Algorithm']",
          "best = fitted[best_name]", "pred = best.predict(X_test)", "", "# ---- رسم نتيجة أفضل موديل ----",
          "import matplotlib.pyplot as plt"]
    if is_cls:
        L += ["cm = skm.confusion_matrix(y_test, pred)",
              "plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap='Greens'); plt.colorbar()",
              "plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)",
              "plt.yticks(range(len(le.classes_)), le.classes_)",
              "for i in range(len(cm)):",
              "    for j in range(len(cm)):",
              "        plt.text(j, i, cm[i, j], ha='center', va='center')",
              "plt.title(f'Confusion Matrix — {best_name}'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()",
              "print(skm.classification_report(y_test, pred, labels=range(len(le.classes_)), "
              "target_names=le.classes_.astype(str), zero_division=0))"]
    else:
        L += ["plt.figure(figsize=(5, 4)); plt.scatter(y_test, pred, s=14, alpha=.7)",
              "lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())",
              "plt.plot([lo, hi], [lo, hi], 'r--'); plt.xlabel('Actual'); plt.ylabel('Predicted')",
              "plt.title(f'Actual vs Predicted — {best_name}'); plt.show()"]
    L += ["", "# ---- Explainability: Permutation Importance (على الأعمدة الأصلية) ----",
          "from sklearn.inspection import permutation_importance",
          f"pi = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=42, scoring='{SCORING[key]}')",
          "imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)",
          "print(imp.head(15).round(4))", "",
          "# ---- حفظ الموديل والتنبؤ بداتا جديدة ----", "import pickle",
          "with open('best_model.pkl', 'wb') as f:",
          "    pickle.dump({'pipeline': best, 'features': FEATURES, "
          + ("'label_encoder': le" if is_cls else "'label_encoder': None") + "}, f)",
          "# new_pred = best.predict(new_df[FEATURES])"]
    return "\n".join(L)


def _load_code(source: str) -> str:
    if source.startswith("sample:"):
        fn = "load_wine" if "wine" in source else "load_diabetes"
        return f"from sklearn.datasets import {fn}\n\ndf = {fn}(as_frame=True).frame\nprint(df.shape)\ndf.head()"
    name = source.split(":", 1)[1]
    read = "pd.read_csv" if name.lower().endswith(".csv") else "pd.read_excel"
    return f"# غيّر المسار لمكان الملف عندك\ndf = {read}({_fmt(name)})\nprint(df.shape)\ndf.head()"


def generate_code(ctx: dict) -> str:
    """ctx: source, task, target, features, clean, config, chosen, params, test_size, do_cv, kinds, metric, imbalance"""
    task, chosen = ctx["task"], ctx["chosen"]
    target = ctx["target"] if is_supervised(task) else None
    parts = [
        ("[markdown]", "# Auto-generated ML script\n# اتولّد من تطبيق No-Code ML — شغّله كما هو أو عدّل فيه."),
        ("", _imports(task, None, chosen, ctx["features"], ctx["config"], ctx["clean"])),
        ("[markdown]", "# 1) تحميل الداتا"),
        ("", _load_code(ctx["source"])),
        ("[markdown]", "# 2) التنضيف (Cleaning + Outliers)"),
        ("", _clean_code(ctx["clean"], target, with_outliers=not is_supervised(task))),
        ("[markdown]", "# 3) التحليل والرسومات (Visualization)"),
        ("", _viz_code(task)),
        ("[markdown]", "# 4) التجهيز (Preprocessing)"),
        ("", _kinds_code(ctx.get("kinds")) + _prep_code(ctx["features"], ctx["config"])),
        ("[markdown]", f"# 5) الموديلات — {TASKS[task]['label']}"),
        ("", _model_code(task, chosen, ctx["params"], ctx["test_size"], ctx["do_cv"],
                         ctx["clean"] if is_supervised(task) else None,
                         metric=ctx.get("metric"), imbalance=ctx.get("imbalance", "none"))),
    ]
    out = []
    for kind, body in parts:
        out.append(f"# %% {kind}".rstrip() + "\n" + body)
    return "\n\n".join(out) + "\n"


def code_to_notebook(code: str) -> bytes:
    cells, cur, kind = [], [], ""
    for line in code.splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append((kind, cur))
            cur, kind = [], "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    if cur:
        cells.append((kind, cur))
    nb_cells = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        if kind == "markdown":
            src = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in src]})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                             "source": [l + "\n" for l in lines]})
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    return json.dumps(nb, ensure_ascii=False, indent=1).encode("utf-8")


# ==================== APP (Streamlit) ====================
import streamlit as st
from sklearn.datasets import load_diabetes, load_wine

st.set_page_config(page_title="ML Studio", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
:root { color-scheme: light; --ink:#16302B; --muted:#5B6F6A; --line:#DCE6E2; --bg:#F4F7F5; --card:#FFFFFF;
        --brand:#0E9F82; --brand-soft:#DDF5EE; --ok:#17734F; --warn:#9A5B00; --bad:#B4303B; --nav:#12302B; }
html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, h5, li, button, input, textarea,
[data-baseweb="select"], [data-baseweb="tab"], [data-testid="stMarkdownContainer"] {
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif !important; }
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], .stAppDeployButton,
[data-testid="stAppHeaderLinks"] { display:none !important; }
[data-testid="stToolbar"] { background: transparent; }
/* زر السايدبار لازم يفضل ظاهر في كل الحالات */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] button,
button[aria-label="Expand sidebar"], button[aria-label="Close sidebar"] {
    display:flex !important; visibility:visible !important; opacity:1 !important; z-index:9999999 !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.6rem 2.2rem 4rem; max-width: 1280px; }
p, li, label p, h1, h2, h3, h4, [data-testid="stCaptionContainer"] { unicode-bidi: plaintext; text-align: start; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--nav); border-right: 0; }
[data-testid="stSidebar"] * { color: #CFE3DC; }
[data-testid="stSidebar"] .brand { display:flex; gap:12px; align-items:center; padding: 6px 4px 18px; }
[data-testid="stSidebar"] .brand .logo { width:38px; height:38px; border-radius:11px; background: var(--brand);
    display:flex; align-items:center; justify-content:center; font-size:20px; }
[data-testid="stSidebar"] .brand b { color:#fff; font-size:17px; display:block; line-height:1.2; }
[data-testid="stSidebar"] .brand span { font-size:12px; color:#8DB0A6; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label { width:100%; padding: 10px 12px; border-radius: 10px; cursor:pointer;
    border-inline-start: 3px solid transparent; transition: background .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: rgba(14,159,130,.28);
    border-inline-start-color: #7FE0C5; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:600; }
[data-testid="stSidebar"] .chips { display:flex; flex-direction:column; gap:6px; margin-top:14px; padding-top:14px;
    border-top: 1px solid rgba(255,255,255,.09); }
[data-testid="stSidebar"] .chip { font-size:12.5px; display:flex; justify-content:space-between; gap:8px; }
[data-testid="stSidebar"] .chip b { color:#fff; font-weight:500; text-align:end; }
[data-testid="stSidebar"] .stButton > button { background: transparent; border: 1px solid rgba(255,255,255,.18); color:#CFE3DC; width:100%; }
[data-testid="stSidebar"] .stButton > button:hover { border-color:#7FE0C5; color:#fff; }

/* ---------- page header / cards ---------- */
.ph { margin: 0 0 18px; }
.ph h1 { font-size: 26px; font-weight: 700; margin: 0 0 4px; padding: 0; color: var(--ink); }
.ph p { margin: 0; color: var(--muted); font-size: 14.5px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border: 1px solid var(--line) !important;
    border-radius: 14px !important; }
.ct { font-size: 15.5px; font-weight: 600; margin: 2px 0 2px; color: var(--ink); }
.cs { font-size: 13px; color: var(--muted); margin: 0 0 10px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
.kpi .l { font-size: 12.5px; color: var(--muted); }
.kpi .v { font-size: 24px; font-weight: 700; color: var(--ink); line-height: 1.25; }
.kpi.warn .v { color: var(--warn); } .kpi.bad .v { color: var(--bad); } .kpi.ok .v { color: var(--ok); }
.ins { margin: 6px 0; padding: 9px 12px; border-radius: 9px; background:#F4F8F6; border-inline-start: 4px solid #7C8F8A; font-size: 14px; }
.ins.warn { border-inline-start-color:#F2994A; background:#FFF8EC; } .ins.ok { border-inline-start-color:#1F9D6B; background:#EEF9F3; }
.badge { display:inline-block; padding: 2px 10px; border-radius: 99px; font-size: 12px; background: var(--brand-soft); color: var(--brand); font-weight:600; }
.empty { text-align:center; padding: 54px 10px; color: var(--muted); }
.empty h3 { color: var(--ink); margin-bottom: 4px; }

/* ---------- controls ---------- */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; border: 1px solid var(--line); padding: .5rem 1.1rem; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); color:#fff; }
.stButton > button[kind="primary"]:hover { background:#0B8670; border-color:#0B8670; }
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--brand); }
[data-baseweb="tab-highlight"] { background-color: var(--brand) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; }
[data-testid="stProgress"] > div > div > div { background-color: var(--brand); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PAGES = [("data", "البيانات"), ("eda", "التحليل"), ("clean", "التنضيف"), ("prep", "التجهيز"),
         ("algo", "الخوارزميات"), ("res", "النتائج"), ("code", "الكود")]
PAGE_INFO = {
    "data": ("البيانات", "ارفع ملف أو جرّب داتا جاهزة، وحدد عايز تعمل إيه بالظبط."),
    "eda": ("تحليل الداتا", "افهم الداتا قبل ما تبدأ: المشاكل، التوزيعات، الارتباطات — وحمّل تقرير جاهز."),
    "clean": ("التنضيف و الـ Outliers", "شيل المكرر، عالج القيم الشاذة، وشوف الفرق قبل وبعد."),
    "prep": ("التجهيز", "راجع أنواع الأعمدة، اختار الـ Features وطريقة تجهيز كل عمود (Missing / Scaling / Encoding)."),
    "algo": ("الخوارزميات", "اختار الخوارزميات والمقياس ومعالجة عدم التوازن، وبعدين درّب."),
    "res": ("النتائج", "قارن الموديلات، افهم قراراتها (Explainability)، وحمّل الناتج."),
    "code": ("الكود", "الكود الكامل لكل اللي عملته (تنضيف ← رسومات ← تجهيز ← موديل) جاهز تشغّله برا التطبيق."),
}

S = st.session_state.setdefault("S", {})


# ---------- widgets بتحفظ قيمتها في S (فمتضيعش لما تتنقل بين الصفحات) ----------
def _sync(key):
    def cb():
        S[key] = st.session_state["w_" + key]
    return cb


def w_radio(key, label, options, default=None, fmt=str, horizontal=True, collapsed=True):
    cur = S.get(key, options[0] if default is None else default)
    cur = cur if cur in options else options[0]
    S[key] = st.session_state["w_" + key] = cur
    st.radio(label, options, key="w_" + key, format_func=fmt, horizontal=horizontal, on_change=_sync(key),
             label_visibility="collapsed" if collapsed else "visible")
    return S[key]


def w_select(key, label, options, default=None, fmt=str, help=None):
    cur = S.get(key, options[0] if default is None else default)
    cur = cur if cur in options else options[0]
    S[key] = st.session_state["w_" + key] = cur
    st.selectbox(label, options, key="w_" + key, format_func=fmt, on_change=_sync(key), help=help)
    return S[key]


def w_multi(key, label, options, default=None, with_all=True):
    """Multiselect + checkbox «اختيار الكل»."""
    options = list(options)
    cur = S.get(key, list(options) if default is None else default)
    cur = [o for o in cur if o in options]
    S[key] = cur
    if with_all:
        ak = "all_" + key

        def toggle():
            S[key] = list(options) if st.session_state[ak] else []

        st.session_state[ak] = bool(options) and len(cur) == len(options)
        st.checkbox(f"اختيار الكل ({len(options)})", key=ak, on_change=toggle)
    st.session_state["w_" + key] = S[key]
    st.multiselect(label, options, key="w_" + key, on_change=_sync(key))
    return S[key]


def w_slider(key, label, lo, hi, default, step=None, help=None):
    is_int = all(isinstance(v, int) for v in (lo, hi, default))
    cast = int if is_int else float
    cur = min(max(cast(S.get(key, default)), cast(lo)), cast(hi))
    S[key] = st.session_state["w_" + key] = cur
    kw = {"step": cast(step)} if step is not None else {}
    st.slider(label, cast(lo), cast(hi), key="w_" + key, on_change=_sync(key), help=help, **kw)
    return S[key]


def w_check(key, label, default=False, help=None):
    S[key] = st.session_state["w_" + key] = bool(S.get(key, default))
    st.checkbox(label, key="w_" + key, on_change=_sync(key), help=help)
    return S[key]


def w_toggle(key, label, default=False, help=None):
    S[key] = st.session_state["w_" + key] = bool(S.get(key, default))
    st.toggle(label, key="w_" + key, on_change=_sync(key), help=help)
    return S[key]


def w_text(key, label, default="", help=None):
    S[key] = st.session_state["w_" + key] = str(S.get(key, default))
    st.text_input(label, key="w_" + key, on_change=_sync(key), help=help)
    return S[key]


# ---------- عناصر شكلية ----------
def show_df(df, height=None, hide_index=False):
    kw = {"hide_index": hide_index}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def page_header(key):
    t, d = PAGE_INFO[key]
    st.markdown(f"<div class='ph'><h1>{t}</h1><p>{d}</p></div>", unsafe_allow_html=True)


def card_title(title, sub=""):
    st.markdown(f"<div class='ct'>{title}</div>" + (f"<div class='cs'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


def kpis(items):
    """items: [(label, value, tone)]"""
    cols = st.columns(len(items))
    for col, (l, v, *tone) in zip(cols, items):
        t = tone[0] if tone else ""
        col.markdown(f"<div class='kpi {t}'><div class='l'>{l}</div><div class='v'>{v}</div></div>",
                     unsafe_allow_html=True)


def insight_list(items):
    st.markdown("".join(f"<div class='ins {lvl}'>{txt}</div>" for lvl, txt in items), unsafe_allow_html=True)


def goto(page):
    S["page"] = page


def nav_row(prev=None, nxt=None, next_label="التالي", disabled=False):
    st.write("")
    c1, c2, c3 = st.columns([1, 4, 1])
    if prev:
        c1.button("السابق", key=f"prev_{prev}", on_click=goto, args=(prev,))
    if nxt:
        c3.button(next_label, key=f"next_{nxt}", type="primary", on_click=goto, args=(nxt,), disabled=disabled)


def empty_state(title, text, to="data", label="ابدأ من هنا"):
    st.markdown(f"<div class='empty'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)
    _, c, _ = st.columns([2, 1, 2])
    c.button(label, key=f"empty_{to}", type="primary", on_click=goto, args=(to,))


# ---------- الداتا والإعدادات المشتقة ----------
def reset_all():
    S.clear()


def load_dataset(df, name, source):
    keep = {k: S[k] for k in ("page", "src", "sample", "saved_file") if k in S}
    S.clear()
    S.update(keep)
    df = df.copy()
    seen, cols = {}, []
    for c in df.columns:
        c = str(c).strip() or "col"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols
    S.update(raw=df, name=name, source=source)


def read_table(name, raw):
    try:
        if name.lower().endswith(".csv"):
            for enc in ("utf-8-sig", "cp1256", "latin-1"):
                try:
                    return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                except UnicodeDecodeError:
                    continue
        return pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        st.error(f"مقدرتش أقرا الملف: {e}")
        return None


def guess_task(s: pd.Series) -> str:
    return "regression" if (column_kind(s) == "numeric" and s.nunique() > 15) else "classification"


def clean_cfg() -> dict:
    cfg = {k: S.get("c_" + k, v) for k, v in DEFAULT_CLEAN.items()}
    cfg["drop_cols"] = list(cfg["drop_cols"])
    return cfg


@st.cache_data(show_spinner=False)
def cached_clean(df, cfg_json, target):
    return clean_data(df, json.loads(cfg_json), target)


def apply_kinds(df):
    """أنواع الأعمدة اللي المستخدم غيّرها يدويًا (numeric / categorical) — الـ drop بيتعمل في get_features."""
    out = df.copy()
    for c in out.columns:
        k = S.get(f"kind_{c}", "auto")
        if k == "numeric":
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif k == "categorical":
            out[c] = out[c].map(_to_cat_str).astype(object)
    return out


def current_task() -> str:
    return S.get("task", "classification")


def current_target():
    raw = S.get("raw")
    if raw is None:
        return None
    t = S.get("target")
    return t if t in raw.columns else raw.columns[-1]


def get_clean():
    raw = S["raw"]
    target = current_target() if is_supervised(current_task()) else None
    df, log = cached_clean(raw, json.dumps(clean_cfg(), sort_keys=True, default=str), target)
    return apply_kinds(df), log


def get_clean_pre():
    """نفس التنضيف بس من غير الـ Outliers (بتتعمل بعد الـ split وقت التدريب)."""
    raw = S["raw"]
    target = current_target() if is_supervised(current_task()) else None
    cfg = {**clean_cfg(), "outlier_method": "none"}
    return apply_kinds(cached_clean(raw, json.dumps(cfg, sort_keys=True, default=str), target)[0])


def get_features(df):
    """(المرشحين, المختارين). الـ IDs والأعمدة الثابتة بتتشال أوتوماتيك لحد ما المستخدم يختار بنفسه."""
    sup = is_supervised(current_task())
    tgt = current_target() if sup else None
    cands = [c for c in df.columns if c != tgt and S.get(f"kind_{c}", "auto") != "drop"]
    sel = S.get("features")
    if sel is None:
        skip = set(auto_excluded(df, cands))
        return cands, [c for c in cands if c not in skip]
    return cands, [c for c in sel if c in cands]


def get_prep_config(df, features):
    sup = is_supervised(current_task())
    X = prepare_features(df, features)
    auto = S.get("prep_auto", True)
    manual = {}
    for c in features:
        kind = column_kind(X[c])
        opts = IMPUTE_NUM if kind == "numeric" else IMPUTE_CAT
        m = {"impute": S.get(f"imp_{c}"), "scale": S.get(f"sc_{c}"), "encode": S.get(f"enc_{c}")}
        if m["impute"] not in opts:
            m["impute"] = None
        if m["scale"] not in SCALERS:
            m["scale"] = None
        if kind == "categorical" and m["encode"] not in ENCODERS:
            m["encode"] = None
        if kind == "numeric":
            m["encode"] = None
        manual[c] = m
    return X, build_config(X, auto, manual, sup)


def get_params(task, name) -> dict:
    out = {}
    for p, spec in get_registry(task)[name]["params"].items():
        out[p] = S.get(f"hp_{task}_{name}_{p}", default_params(task, name)[p])
    return out


def get_chosen(task):
    reg = list(get_registry(task))
    return [n for n in S.get(f"chosen_{task}", reg[:3]) if n in reg]


def code_context():
    df, _ = get_clean()
    task = current_task()
    _, feats = get_features(df)
    _, config = get_prep_config(df, feats)
    chosen = get_chosen(task)
    tgt = current_target()
    cls_ = task == "classification"
    return {"source": S.get("source", "file:data.csv"), "task": task, "target": tgt, "features": feats,
            "clean": clean_cfg(), "config": config, "chosen": chosen,
            "params": {n: get_params(task, n) for n in chosen},
            "test_size": S.get("test_size", 0.2), "do_cv": S.get("do_cv", True),
            "kinds": {c: S[f"kind_{c}"] for c in df.columns if S.get(f"kind_{c}", "auto") in ("numeric", "categorical")},
            "metric": S.get(f"metric_{tgt}") if cls_ else None,
            "imbalance": S.get(f"imbalance_{tgt}", "none") if cls_ else "none"}


# ---------- رسومات (بتتحول PNG وبتتخزّن في الكاش) ----------
def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def show_fig(fig):
    st.image(fig_png(fig))


@st.cache_data(show_spinner=False, max_entries=48)
def plot_png(kind, df, cols=(), extra=""):
    cols = list(cols)
    fig = {"missing": lambda: plot_missing(df), "hist": lambda: plot_hist_grid(df, cols),
           "box": lambda: plot_box_grid(df, cols), "corr": lambda: plot_corr(df, cols),
           "cat": lambda: plot_cat_counts(df, cols),
           "target": lambda: plot_target(df, extra.split("|")[1], extra.split("|")[0])}[kind]()
    return fig_png(fig)


def show_plot(kind, df, cols=(), extra=""):
    st.image(plot_png(kind, df, tuple(cols), extra))


def plot_scores(names, vals, label):
    fig, ax = _fig1(6, max(1.8, 0.38 * len(names) + 0.8))
    order = np.argsort(vals)
    best = names[int(np.nanargmax(vals))]
    ax.barh([names[i] for i in order], [vals[i] for i in order],
            color=[PALETTE[0] if names[i] == best else SOFT for i in order])
    ax.set_xlabel(label, color=MUTED, fontsize=8)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ==================== SIDEBAR ====================
def sidebar():
    with st.sidebar:
        st.markdown("<div class='brand'><div class='logo'>🧠</div><div><b>ML Studio</b>"
                    "<span>تعلّم آلة من غير كود</span></div></div>", unsafe_allow_html=True)
        keys, labels = [k for k, _ in PAGES], dict(PAGES)
        S["page"] = st.session_state["w_page"] = S.get("page", "data") if S.get("page") in keys else "data"
        st.radio("التنقل", keys, key="w_page", on_change=_sync("page"), label_visibility="collapsed",
                 format_func=lambda k: f"{keys.index(k) + 1}.   {labels[k]}")
        if S.get("raw") is not None:
            raw, task = S["raw"], current_task()
            chips = [("الداتا", str(S.get("name", "-"))[:22]), ("الحجم", f"{raw.shape[0]:,} × {raw.shape[1]}"),
                     ("المهمة", TASKS[task]["label"].split(" (")[0])]
            if is_supervised(task):
                chips.append(("Target", str(current_target())[:22]))
            if S.get("res"):
                chips.append(("التدريب", "تم ✓"))
            st.markdown("<div class='chips'>" + "".join(f"<div class='chip'><span>{a}</span><b>{b}</b></div>"
                                                        for a, b in chips) + "</div>", unsafe_allow_html=True)
        st.write("")
        st.button("بدء من جديد", on_click=reset_all)


# ==================== 1) البيانات ====================
SAMPLES = {"Wine — تصنيف": ("wine", load_wine), "Diabetes — تنبؤ رقمي": ("diabetes", load_diabetes)}


def page_data():
    page_header("data")
    with st.container(border=True):
        card_title("مصدر الداتا")
        src = w_radio("src", "المصدر", ["upload", "saved", "sample"],
                      fmt=lambda v: {"upload": "رفع ملف", "saved": "ملفات سابقة", "sample": "داتا تجريبية"}[v])
        if src == "upload":
            up = st.file_uploader("CSV أو Excel", type=["csv", "xlsx", "xls"], key="uploader")
            if up is not None and S.get("up_sig") != (up.name, up.size):
                d = read_table(up.name, up.getvalue())
                if d is not None:
                    fname = os.path.basename(up.name)
                    with open(os.path.join(UPLOAD_DIR, fname), "wb") as f:   # بنحفظها في السجل
                        f.write(up.getvalue())
                    load_dataset(d, fname, f"file:{fname}")
                    S["up_sig"] = (up.name, up.size)
            if up is None and S.get("raw") is None:
                st.caption("اسحب الملف هنا أو اضغط Browse — الأعمدة النصية والرقمية بتتعرف تلقائيًا.")
        elif src == "saved":
            files = sorted(f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".csv", ".xlsx", ".xls")))
            if not files:
                st.info("مفيش ملفات اتحفظت لسه — ارفع ملف الأول.")
            else:
                fname = w_select("saved_file", "اختار ملف من الملفات السابقة", files)
                if S.get("source") != f"file:{fname}":
                    with open(os.path.join(UPLOAD_DIR, fname), "rb") as f:
                        d = read_table(fname, f.read())
                    if d is not None:
                        load_dataset(d, fname, f"file:{fname}")
        else:
            choice = w_select("sample", "داتا جاهزة", list(SAMPLES))
            sid, loader = SAMPLES[choice]
            if S.get("source") != f"sample:{sid}":
                load_dataset(loader(as_frame=True).frame, choice, f"sample:{sid}")
                S["sample"] = choice
                S["task"] = "classification" if sid == "wine" else "regression"
                S["target"] = "target"

    df = S.get("raw")
    if df is None:
        return
    st.write("")
    nums = numeric_cols(df)
    miss, dup = int(df.isna().sum().sum()), int(df.duplicated().sum())
    kpis([("صفوف", f"{df.shape[0]:,}"), ("أعمدة", df.shape[1]), ("أعمدة رقمية", len(nums)),
          ("Missing", f"{miss:,}", "warn" if miss else "ok"), ("صفوف مكررة", dup, "warn" if dup else "ok")])
    st.write("")
    t1, t2 = st.tabs(["معاينة الداتا", "ملخص الأعمدة"])
    with t1:
        show_df(df.head(50), height=300)
    with t2:
        show_df(column_summary(df), height=300)

    st.write("")
    with st.container(border=True):
        card_title("عايز تعمل إيه بالداتا دي؟", "التصنيف والتنبؤ محتاجين Target. التجميع وكشف الشواذ وتقليل الأبعاد من غير Target.")
        task = w_radio("task", "المهمة", list(TASKS), fmt=lambda t: TASKS[t]["label"], horizontal=False)
        sup = is_supervised(task)
        if S.get("_sup_prev") != sup:
            S["_sup_prev"] = sup
            S.pop("features", None)
        if sup:
            target = w_select("target", "العمود المطلوب التنبؤ به (Target)", list(df.columns), default=df.columns[-1])
            if S.get("_guess_for") != (S.get("name"), target):
                S["_guess_for"] = (S.get("name"), target)
                S.pop("features", None)
                g = guess_task(df[target])
                if g != task and task in ("classification", "regression"):
                    S["task"] = g
                    st.rerun()
            st.caption(f"عدد القيم المختلفة في «{target}»: {df[target].nunique()}")
    nav_row(None, "eda")


# ==================== 2) التحليل ====================
def page_eda():
    page_header("eda")
    raw = S["raw"]
    clean_df, _ = get_clean()
    task = current_task()
    target = current_target() if is_supervised(task) else None
    ver = w_radio("eda_ver", "الداتا", ["raw", "clean"],
                  fmt=lambda v: {"raw": "الداتا الأصلية", "clean": "بعد التنضيف"}[v])
    df = raw if ver == "raw" else clean_df
    nums = numeric_cols(df)
    cats = [c for c in df.columns if c not in nums]
    miss, dup = int(df.isna().sum().sum()), int(df.duplicated().sum())
    st.write("")
    kpis([("صفوف", f"{df.shape[0]:,}"), ("أعمدة", df.shape[1]), ("رقمية / نصية", f"{len(nums)} / {len(cats)}"),
          ("Missing", f"{miss:,}", "warn" if miss else "ok"), ("مكرر", dup, "warn" if dup else "ok")])
    st.write("")

    with st.container(border=True):
        card_title("ملاحظات تلقائية على الداتا")
        insight_list(data_insights(df, target))

    st.write("")
    with st.container(border=True):
        card_title("التقرير", "شيت Excel (ملخص، أعمدة، إحصائيات، Outliers، ارتباط) + تقرير HTML فيه الرسومات.")
        rkey = (ver, S.get("name"), df.shape)
        if st.button("إنشاء الشيت والتقرير"):
            with st.spinner("بيجهّز التقرير…"):
                S["report"] = {"key": rkey, "xlsx": build_report_xlsx(df, target),
                               "html": build_report_html(df, target, f"Data Report — {S.get('name', '')}")}
        rep = S.get("report")
        if rep and rep["key"] == rkey:
            c1, c2, _ = st.columns([1, 1, 2])
            if rep["xlsx"]:
                c1.download_button("تحميل Excel", rep["xlsx"], file_name="data_report.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                c1.caption("محتاج `pip install openpyxl` لتصدير Excel.")
            c2.download_button("تحميل HTML", rep["html"], file_name="data_report.html", mime="text/html")

    st.write("")
    tabs = st.tabs(["نظرة عامة", "القيم الناقصة", "التوزيعات", "Outliers", "الارتباط", "Target", "الأعمدة النصية"])
    with tabs[0]:
        show_df(column_summary(df), height=280)
        if nums:
            show_df(df[nums].describe().T.round(3), height=280)
    with tabs[1]:
        show_plot("missing", df)
    with tabs[2]:
        if nums:
            cols = w_multi("eda_hist", "الأعمدة (أقصى 12 رسم)", nums, default=nums[:6])
            if cols:
                show_plot("hist", df, cols[:12])
        else:
            st.info("مفيش أعمدة رقمية.")
    with tabs[3]:
        if nums:
            show_df(count_outliers(df, nums).sort_values("Outliers", ascending=False), height=240, hide_index=True)
            cols = w_multi("eda_box", "أعمدة الـ Boxplot", nums, default=nums[:8])
            if cols:
                show_plot("box", df, cols[:12])
            st.caption("النقط الحمرا برا الصندوق = قيم شاذة. عالجها من صفحة التنضيف.")
        else:
            st.info("مفيش أعمدة رقمية.")
    with tabs[4]:
        if len(nums) > 1:
            show_plot("corr", df, nums[:25])
            corr = df[nums[:60]].corr().abs()
            pairs = [{"A": a, "B": b, "|r|": round(corr.loc[a, b], 3)} for i, a in enumerate(corr.columns)
                     for b in corr.columns[i + 1:]]
            if pairs:
                show_df(pd.DataFrame(pairs).sort_values("|r|", ascending=False).head(10), hide_index=True)
        else:
            st.info("محتاج عمودين رقميين على الأقل.")
    with tabs[5]:
        if target:
            show_plot("target", df, extra=f"{task}|{target}")
        else:
            st.info("المهمة المختارة من غير Target.")
    with tabs[6]:
        cc = [c for c in cats if df[c].nunique() <= 30]
        if cc:
            show_plot("cat", df, cc[:9])
        else:
            st.info("مفيش أعمدة نصية مناسبة للرسم.")
    nav_row("data", "clean")


# ==================== 3) التنضيف ====================
def page_clean():
    page_header("clean")
    raw = S["raw"]
    task = current_task()
    target = current_target() if is_supervised(task) else None
    left, right = st.columns(2)
    with left, st.container(border=True):
        card_title("التنضيف الأساسي")
        w_toggle("c_fix_types", "حوّل الأعمدة النصية اللي أغلبها أرقام", True)
        w_toggle("c_drop_dupes", "احذف الصفوف المكررة", True)
        w_slider("c_max_missing_pct", "احذف العمود لو الـ Missing فيه أكتر من (%) — 100 = إيقاف", 0, 100, 60, step=5)
        w_multi("c_drop_cols", "أعمدة تحذفها بإيدك (اختياري)", [c for c in raw.columns if c != target], default=[])
    numerics = numeric_cols(raw, exclude=[target] if target else [])
    with right, st.container(border=True):
        card_title("معالجة الـ Outliers", "القيم الشاذة بتأثر على الموديلات، خصوصًا الخطية و KNN. "
                                          "في التدريب بتتعالج بعد الـ split (الحدود من الـ Train بس).")
        method = w_radio("c_outlier_method", "الطريقة", list(OUTLIER_METHODS), fmt=lambda m: OUTLIER_METHODS[m],
                         horizontal=False)
        if method != "none":
            if method in ("iqr", "zscore"):
                w_select("c_outlier_action", "تعمل إيه في القيمة الشاذة؟", list(OUTLIER_ACTIONS),
                         fmt=lambda a: OUTLIER_ACTIONS[a])
                if method == "iqr":
                    w_slider("c_iqr_k", "معامل IQR (أقل = أشد)", 1.0, 4.0, 1.5, step=0.1)
                else:
                    w_slider("c_z_thr", "حد Z-Score", 2.0, 5.0, 3.0, step=0.1)
            else:
                w_slider("c_contamination", "نسبة الشواذ المتوقعة", 0.01, 0.3, 0.05, step=0.01)
            w_multi("c_outlier_cols", "الأعمدة اللي تتعالج", numerics)
            if not numerics:
                st.warning("مفيش أعمدة رقمية تتعالج.")

    clean_df, log = get_clean()
    st.write("")
    with st.container(border=True):
        card_title("النتيجة")
        kpis([("صفوف", f"{raw.shape[0]:,} ← {clean_df.shape[0]:,}"), ("أعمدة", f"{raw.shape[1]} ← {clean_df.shape[1]}"),
              ("Missing", f"{int(raw.isna().sum().sum()):,} ← {int(clean_df.isna().sum().sum()):,}")])
        st.write("")
        insight_list([("info", t) for t in log] or [("ok", "مفيش تغييرات اتطبقت — الداتا زي ما هي.")])
        cols_now = [c for c in numerics if c in clean_df.columns and column_kind(clean_df[c]) == "numeric"]
        if cols_now:
            pick = w_multi("c_cmp_cols", "قارن قبل وبعد (Boxplot)", cols_now, default=cols_now[:4], with_all=False)
            if pick:
                a, b = st.columns(2)
                a.caption("قبل")
                b.caption("بعد")
                with a:
                    show_plot("box", raw, pick[:8])
                with b:
                    show_plot("box", clean_df, pick[:8])
        with st.expander("معاينة الداتا بعد التنضيف"):
            show_df(clean_df.head(30), height=260)
    nav_row("eda", "prep")


# ==================== 4) التجهيز ====================
def apply_bulk(items):
    S["prep_auto"] = False
    for c, kind in items:
        S[f"imp_{c}"] = S.get("bulk_imp", "median") if kind == "numeric" else "most_frequent"
        S[f"sc_{c}"] = S.get("bulk_sc", "Standard")
        if kind == "categorical":
            S[f"enc_{c}"] = S.get("bulk_enc", "onehot")


def drop_suspicious(sus, cands):
    S["features"] = [c for c in S.get("features", cands) if c not in sus]


def page_prep():
    page_header("prep")
    df, _ = get_clean()
    task = current_task()
    sup = is_supervised(task)
    target = current_target() if sup else None

    with st.container(border=True):
        card_title("أنواع الأعمدة", "راجعها قبل ما تبدأ: مثلاً ZipCode أو تقييم من 1 لـ 5 لازم يبقوا categorical، "
                                    "والـ ID يتشال. auto = التطبيق يقرر.")
        with st.expander("عدّل أنواع الأعمدة (auto / numeric / categorical / drop)"):
            for c in [c for c in df.columns if c != target]:
                k1, k2 = st.columns([2, 1])
                k1.markdown(f"**{c}**  \n<span class='badge'>{column_kind(df[c])}</span> "
                            f"· {df[c].nunique(dropna=True)} قيمة مختلفة"
                            + (" · <b>ID؟</b>" if is_id_like(df[c]) else ""), unsafe_allow_html=True)
                with k2:
                    w_select(f"kind_{c}", "النوع", KIND_OPTS, "auto")
    df, _ = get_clean()      # بعد تطبيق الأنواع
    cands, default_feats = get_features(df)[0], None
    skip = set(auto_excluded(df, cands))
    default_feats = [c for c in cands if c not in skip]

    st.write("")
    with st.container(border=True):
        card_title("الأعمدة المستخدمة (Features)")
        w_multi("features", "الأعمدة", cands, default=default_feats)
        sus = [c for c in cands if c in skip]
        if sus and any(c in S.get("features", cands) for c in sus):
            st.warning("أعمدة ID أو ثابتة (مالهاش فايدة وممكن تضلل الموديل): " + ", ".join(map(str, sus)))
            st.button("استبعد المشبوهة", on_click=drop_suspicious, args=(sus, cands))
    _, feats = get_features(df)
    if not feats:
        st.warning("اختار عمود واحد على الأقل.")
        nav_row("clean", None)
        return
    st.write("")
    X = prepare_features(df, feats)
    enc_opts = ENCODERS if sup else [e for e in ENCODERS if e != "target"]
    with st.container(border=True):
        card_title("تجهيز الأعمدة", "Missing values + Scaling + Encoding لكل عمود. كله بيتعمل fit على الـ Train بس (من غير Data Leakage).")
        auto = w_toggle("prep_auto", "Auto — خلّي التطبيق يختار الأنسب لكل عمود", True)
        if auto:
            st.caption("الـ Auto: Missing كتير ← Missing-Indicator / فئة «missing» · فئات أقل من 11 ← One-Hot · "
                       "فئات كتير ← Target Encoding (بدل One-Hot اللي بيفجّر الميموري).")
        if not auto:
            with st.expander("إعداد جماعي — طبّق اختيار واحد على كل الأعمدة"):
                b1, b2, b3 = st.columns(3)
                with b1:
                    w_select("bulk_imp", "Missing (للأرقام)", IMPUTE_NUM, "median")
                with b2:
                    w_select("bulk_sc", "Scaler", list(SCALERS), "Standard")
                with b3:
                    w_select("bulk_enc", "Encoder (للنصوص)", enc_opts, "onehot")
                st.button("طبّق على الكل", on_click=apply_bulk, args=([(c, column_kind(X[c])) for c in feats],))
            for c in feats:
                kind, a = column_kind(X[c]), auto_config(X[c], sup)
                c0, c1, c2, c3 = st.columns([1.3, 1, 1, 1])
                c0.markdown(f"**{c}**  \n<span class='badge'>{kind}</span>", unsafe_allow_html=True)
                with c1:
                    w_select(f"imp_{c}", "Missing", IMPUTE_NUM if kind == "numeric" else IMPUTE_CAT, a["impute"])
                with c2:
                    w_select(f"sc_{c}", "Scaler", list(SCALERS), a["scale"])
                with c3:
                    if kind == "categorical":
                        w_select(f"enc_{c}", "Encoder", enc_opts, a["encode"])
                    else:
                        st.caption("— (رقمي)")
    _, config = get_prep_config(df, feats)
    summary = pd.DataFrame([{"العمود": c, "النوع": config[c]["kind"], "Missing": config[c]["impute"],
                             "Missing-Flag": "✓" if config[c].get("indicator") else "-",
                             "Scale": config[c]["scale"], "Encode": config[c]["encode"] or "-"} for c in feats])
    with st.expander("ملخص التجهيز النهائي", expanded=auto):
        show_df(summary, height=260, hide_index=True)
    with st.expander("معاينة الداتا بعد التجهيز (أول 10 صفوف)"):
        try:
            pre = build_preprocessor(X, config)
            Xh = X.head(500)
            yv = encoded_target(df, target, task).loc[Xh.index] if sup else None
            Zs = pre.fit_transform(Xh, yv)
            show_df(pd.DataFrame(Zs, columns=clean_feature_names(pre)).head(10).round(3))
            st.caption(f"عدد الأعمدة بعد التجهيز: {Zs.shape[1]}")
            if Zs.shape[1] > 300:
                st.warning("عدد الأعمدة بعد التجهيز كبير — راجع الأعمدة اللي فيها فئات كتير.")
        except Exception as e:
            st.error(f"فيه مشكلة في التجهيز: {e}")
    nav_row("clean", "algo")


# ==================== 5) الخوارزميات ====================
def hp_widget(task, name, p, spec):
    key = f"hp_{task}_{name}_{p}"
    kind = spec[0]
    if kind == "int":
        w_slider(key, p, int(spec[1]), int(spec[2]), int(spec[3]))
    elif kind == "float":
        lo, hi, d = float(spec[1]), float(spec[2]), float(spec[3])
        w_slider(key, p, lo, hi, d, step=max(round((hi - lo) / 200, 4), 0.0001))
    elif kind == "choice":
        w_select(key, p, spec[1], spec[2])
    else:
        w_text(key, p, spec[1], help="أرقام مفصولة بفاصلة — مثال: 128,64,32 = تلات طبقات مخفية")


def run_training(df, task, chosen, feats, config):
    bar = st.progress(0.0)
    params = {n: get_params(task, n) for n in chosen}
    max_rows = int(S.get("max_rows", 50000))
    for k in [k for k in S if str(k).startswith(("perm_", "shap_"))]:
        S.pop(k)
    with st.spinner("بيدرّب الموديلات…"):
        if is_supervised(task):
            ocfg = clean_cfg()
            tgt = current_target()
            # لو في معالجة Outliers: ندرّب على الداتا قبلها، والمعالجة بتتم بعد الـ split (من غير leakage)
            base = get_clean_pre() if ocfg["outlier_method"] != "none" else df
            cls_ = task == "classification"
            res = train_supervised(cap_rows(base, max_rows), tgt, feats, task, chosen, params, config,
                                   S.get("test_size", 0.2), S.get("do_cv", True), progress=bar.progress,
                                   outlier_cfg=ocfg, metric=S.get(f"metric_{tgt}") if cls_ else None,
                                   imbalance=S.get(f"imbalance_{tgt}", "none") if cls_ else "none")
        else:
            d = cap_rows(df, max_rows)
            res = train_unsupervised(d, feats, task, chosen, params, config, progress=bar.progress)
            res["data"] = d[feats].copy()
    S["res"] = res
    S["page"] = "res"
    st.rerun()


def page_algo():
    page_header("algo")
    df, _ = get_clean()
    task = current_task()
    reg = get_registry(task)
    _, feats = get_features(df)
    if not feats:
        empty_state("مفيش أعمدة مختارة", "ارجع لصفحة التجهيز واختار Features.", "prep", "روح للتجهيز")
        return
    _, config = get_prep_config(df, feats)
    with st.container(border=True):
        card_title(f"الخوارزميات — {TASKS[task]['label']}", "اختار واحدة أو أكتر، أو اختار الكل وقارن.")
        chosen = w_multi(f"chosen_{task}", "الخوارزميات", list(reg), default=list(reg)[:3])
        with_params = [n for n in chosen if reg[n]["params"]]
        if with_params:
            with st.expander("الإعدادات (Hyperparameters) — اختياري"):
                for n in with_params:
                    st.markdown(f"**{n}**")
                    items = list(reg[n]["params"].items())
                    cols = st.columns(min(len(items), 3))
                    for i, (p, spec) in enumerate(items):
                        with cols[i % len(cols)]:
                            hp_widget(task, n, p, spec)

    if is_supervised(task):
        tgt = current_target()
        y = df[tgt]
        st.write("")
        with st.container(border=True):
            card_title("إعدادات التقييم")
            c1, c2 = st.columns(2)
            with c1:
                w_slider("test_size", "نسبة الـ Test", 0.1, 0.5, 0.2, step=0.05)
            with c2:
                st.write("")
                w_check("do_cv", "Cross-validation (5-fold) — أبطأ على الداتا الكبيرة", len(df) <= SLOW_ROWS)
            if task == "classification":
                if y.nunique() < 2:
                    st.error("الـ Target فيه قيمة واحدة بس — مينفعش تصنيف.")
                    return
                imb = is_imbalanced(y)
                if imb:
                    share = y.value_counts(normalize=True)
                    insight_list([("warn", f"الداتا غير متوازنة: أصغر كلاس «{share.idxmin()}» = {100 * share.min():.1f}%. "
                                           "الـ Accuracy هتبان عالية وهي مضللة، فالمقياس الافتراضي اتحول لـ F1 / PR-AUC.")])
                m1, m2 = st.columns(2)
                with m1:
                    w_select(f"metric_{tgt}", "المقياس الأساسي (بيترتّب بيه الموديلات)", metric_options(y),
                             default_metric(y), help="PR-AUC متاح للتصنيف الثنائي بس (محسوب على الكلاس الأقل).")
                with m2:
                    modes = list(IMBALANCE_MODES) if HAS_IMB else ["none", "class_weight"]
                    w_select(f"imbalance_{tgt}", "معالجة عدم التوازن", modes,
                             "class_weight" if imb else "none", fmt=lambda m: IMBALANCE_MODES[m])
                    if not HAS_IMB:
                        st.caption("لتفعيل SMOTE: `pip install imbalanced-learn`")
                if S.get(f"imbalance_{tgt}") == "smote":
                    st.caption("SMOTE بيتطبق جوه الـ Pipeline على الـ Train بس — الـ Test بيفضل حقيقي.")

    st.write("")
    with st.container(border=True):
        card_title("حدود الأداء", "عشان السيرفرات المجانية متقعش: لو الداتا أكبر من الحد ده بيتاخد منها عينة عشوائية للتدريب.")
        max_rows = w_slider("max_rows", "أقصى عدد صفوف للتدريب", 1000, 200000, 50000, step=1000)
        if len(df) > max_rows:
            st.caption(f"هيتدرّب على عينة {max_rows:,} من {len(df):,} صف.")
        used = min(len(df), max_rows)
        slow = [n for n in chosen if n in SLOW_ALGOS]
        if used > SLOW_ROWS and slow:
            st.warning("خوارزميات بطيئة على الحجم ده: " + ", ".join(slow) + " — قلّل الصفوف أو شيلها.")
        if "target" in {config[c].get("encode") for c in feats if config[c]["kind"] == "categorical"} and \
                is_supervised(task):
            st.caption("فيه أعمدة بقيم كتير بتتعمل لها Target Encoding (جوه الـ Pipeline على الـ Train بس).")
    st.write("")
    if st.button("ابدأ التدريب", type="primary", disabled=not chosen):
        run_training(df, task, chosen, feats, config)
    nav_row("prep", "res" if S.get("res") else None, "شوف آخر نتائج")


# ==================== 6) النتائج ====================
def importance_of(pipe):
    m = pipe.named_steps["model"]
    m = getattr(m, "regressor_", m)
    names = clean_feature_names(pipe.named_steps["prep"])
    if hasattr(m, "feature_importances_"):
        v = np.asarray(m.feature_importances_)
    elif hasattr(m, "coef_"):
        v = np.abs(np.asarray(m.coef_))
        v = v.mean(axis=0) if v.ndim > 1 else v
    else:
        return None
    return (names, v) if len(v) == len(names) else None


def permutation_imp(pipe, Xte, yte, scoring):
    """أهمية كل عمود أصلي (من غير ما نتأثر بالـ Encoding) — على الـ Test."""
    from sklearn.inspection import permutation_importance
    Xs, ys = Xte, yte
    if len(Xs) > 1500:
        idx = np.random.RandomState(42).choice(len(Xs), 1500, replace=False)
        Xs, ys = Xs.iloc[idx], ys.iloc[idx]
    try:
        pi = permutation_importance(pipe, Xs, ys, n_repeats=5, random_state=42, scoring=scoring, n_jobs=1)
    except Exception:
        pi = permutation_importance(pipe, Xs, ys, n_repeats=5, random_state=42, n_jobs=1)
    return list(Xs.columns), np.asarray(pi.importances_mean)


def shap_imp(pipe, Xte, max_rows=200):
    """SHAP للموديلات الشجرية (على الأعمدة بعد التجهيز)."""
    import shap
    pre = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    model = getattr(model, "regressor_", model)
    Z = pre.transform(Xte.head(max_rows))
    names = clean_feature_names(pre)
    sv = shap.TreeExplainer(model).shap_values(Z)
    arr = np.abs(np.asarray(sv))
    nf = len(names)
    if arr.ndim == 2:
        vals = arr.mean(axis=0)
    elif arr.shape[-1] == nf:
        vals = arr.mean(axis=tuple(range(arr.ndim - 1)))
    else:
        vals = arr.mean(axis=tuple(a for a in range(arr.ndim) if a != 1))
    return names, np.asarray(vals)


def results_supervised(res):
    task = res["task"]
    key = res.get("key") or ("Accuracy" if task == "classification" else "R2")
    table = res["table"]
    if key in table:
        table = table.sort_values(key, ascending=False, na_position="last")
    ok_models = [n for n in table["Algorithm"] if n in res["fitted"]]
    if not ok_models:
        st.error("كل الموديلات فشلت. شوف عمود Error:")
        show_df(table, hide_index=True)
        return
    best = ok_models[0]
    kpis([("أفضل موديل", best, "ok"), (key, f"{float(table[table.Algorithm == best][key].iloc[0]):.4f}"),
          ("عدد الموديلات", len(ok_models))])
    if res.get("imbalanced"):
        extra = ""
        if res.get("imbalance", "none") != "none":
            extra = f" معالجة التوازن المستخدمة: {IMBALANCE_MODES[res['imbalance']]}."
        insight_list([("warn", f"الداتا غير متوازنة، فالترتيب هنا بـ {key} مش Accuracy (الـ Accuracy ممكن تبان عالية "
                               f"والموديل بيتجاهل الكلاس الصغير).{extra}")])
    if "Fit" in table:
        fit_row = table[table.Algorithm == best].iloc[0]
        msgs = {FIT_FLAGS["over"]: ("warn", f"أفضل موديل ({best}) أداؤه على الـ Train أعلى من الـ Test بفرق {fit_row['Gap']:.2f} — "
                                            "غالباً بيحفظ الداتا. جرّب تقلّل max_depth أو تزوّد الـ regularization (C/alpha) أو تجيب داتا أكتر."),
                FIT_FLAGS["under"]: ("warn", f"أفضل موديل ({best}) أداؤه ضعيف حتى على الـ Train — "
                                             "جرّب موديل أقوى، أو أعمدة أكتر/أنضف، أو تزوّد تعقيد الموديل.")}
        if fit_row["Fit"] in msgs:
            insight_list([msgs[fit_row["Fit"]]])
        else:
            insight_list([("ok", f"أفضل موديل ({best}) متوازن: مفيش فرق كبير بين الـ Train والـ Test.")])
        others = [n for n, f in zip(table.Algorithm, table["Fit"]) if n != best and f == FIT_FLAGS["over"]]
        if others:
            insight_list([("info", "موديلات تانية فيها احتمال Overfitting: " + "، ".join(others))])
    st.write("")
    with st.container(border=True):
        card_title("مقارنة الخوارزميات", f"مرتّبة بـ {key}.")
        try:
            show_df(table.style.format(precision=4, na_rep="-").highlight_max(subset=[key], color="#DDF5EE"),
                    hide_index=True)
        except Exception:
            show_df(table, hide_index=True)
        scores = table[table.Algorithm.isin(ok_models)]
        show_fig(plot_scores(list(scores.Algorithm), list(scores[key].fillna(0)), key))
    st.write("")
    with st.container(border=True):
        pick = w_select("res_pick", "تفاصيل موديل", ok_models)
        pipe = res["fitted"][pick]
        pred = pipe.predict(res["Xte"])
        yte = np.asarray(res["yte"])
        c1, c2 = st.columns(2)
        with c1:
            if task == "classification":
                labels = list(res["le"].classes_)
                show_fig(plot_confusion(skm.confusion_matrix(yte, pred, labels=range(len(labels))), labels))
            else:
                show_fig(plot_actual_pred(yte, pred))
        with c2:
            imp = importance_of(pipe)
            if imp:
                st.caption("أهم الأعمدة بعد التجهيز (Feature importance / |coef|)")
                show_fig(plot_importance(*imp))
            else:
                st.caption("الموديل ده مفيهوش Feature importance مباشر — استخدم Permutation تحت.")
        rep = None
        if task == "classification":
            classes = list(res["le"].classes_)
            rep = pd.DataFrame(skm.classification_report(
                yte, pred, labels=list(range(len(classes))), target_names=[str(c) for c in classes],
                output_dict=True, zero_division=0)).T.round(3)
            with st.expander("Classification report"):
                show_df(rep)

    st.write("")
    with st.container(border=True):
        card_title("ليه الموديل قرر كده؟ (Explainability)",
                   "Permutation Importance: بنبوّظ عمود عمود في الـ Test ونشوف الأداء بيقع قد إيه — بيشتغل مع أي موديل.")
        pk = f"perm_{pick}"
        if st.button("احسب Permutation Importance", key=f"btn_{pk}"):
            with st.spinner("بيحسب…"):
                try:
                    S[pk] = permutation_imp(pipe, res["Xte"], res["yte"], res.get("scoring"))
                except Exception as e:
                    st.error(f"مقدرتش أحسبها: {e}")
        if S.get(pk):
            names, vals = S[pk]
            a, b = st.columns([3, 2])
            with a:
                show_fig(plot_importance(names, vals))
            with b:
                show_df(pd.DataFrame({"Feature": names, "Importance": np.round(vals, 4)})
                        .sort_values("Importance", ascending=False), height=260, hide_index=True)
            st.caption("القيمة الأعلى = العمود أهم للموديل. قيمة قريبة من صفر أو سالبة = العمود مش بيفيد.")
        try:
            import shap  # noqa: F401
            has_shap = True
        except Exception:
            has_shap = False
        if has_shap:
            sk = f"shap_{pick}"
            if st.button("احسب SHAP (للموديلات الشجرية)", key=f"btn_{sk}"):
                with st.spinner("بيحسب SHAP…"):
                    try:
                        S[sk] = shap_imp(pipe, res["Xte"])
                    except Exception as e:
                        st.warning(f"SHAP بيشتغل مع الموديلات الشجرية بس (RandomForest / GradientBoosting / XGBoost…): {e}")
            if S.get(sk):
                st.caption("SHAP — متوسط التأثير المطلق لكل عمود")
                show_fig(plot_importance(*S[sk]))
        else:
            st.caption("لتفعيل SHAP: `pip install shap`")

    st.write("")
    with st.container(border=True):
        card_title("حفظ الموديل في سجل التطبيق")
        model_name_input = st.text_input("اسم الموديل للحفظ:", placeholder="مثال: Random_Forest_Model")
        if st.button("حفظ الموديل بالكامل"):
            safe = re.sub(r"[^\w\-]+", "_", model_name_input.strip(), flags=re.UNICODE).strip("_")
            if safe:
                row = table[table.Algorithm == pick].iloc[0].to_dict()
                metrics = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
                           for k, v in row.items() if pd.notna(v)}
                with open(os.path.join(MODELS_DIR, f"{safe}.pkl"), "wb") as f:
                    pickle.dump({"pipeline": pipe, "features": res["features"], "label_encoder": res["le"],
                                 "task": task}, f)
                meta = {"model_name": safe, "algorithm": pick, "task": task, "features": list(res["features"]),
                        "main_metric": key, "metrics": metrics}
                with open(os.path.join(MODELS_DIR, f"{safe}_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=4)
                st.success(f"تم حفظ الموديل '{safe}' بنجاح! 🎯")
            else:
                st.warning("اكتب اسم للموديل الأول.")
        out = res["Xte"].copy()
        out["actual"] = res["le"].inverse_transform(yte) if res["le"] is not None else yte
        out["predicted"] = res["le"].inverse_transform(pred) if res["le"] is not None else pred
        d1, d2, _ = st.columns([1, 1, 2])
        blob = pickle.dumps({"pipeline": pipe, "features": res["features"], "label_encoder": res["le"],
                             "task": task})
        d1.download_button("تحميل الموديل (.pkl)", blob, file_name=f"{pick}.pkl")
        d2.download_button("تحميل التوقعات (.csv)", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="predictions.csv", mime="text/csv")


def results_unsupervised(res):
    task, table, outs = res["task"], res["table"], res["outputs"]
    ok = [n for n in table["Algorithm"] if n in outs]
    if not ok:
        st.error("كل الخوارزميات فشلت:")
        show_df(table, hide_index=True)
        return
    if task == "clustering" and "Silhouette" in table:
        table = table.sort_values("Silhouette", ascending=False, na_position="last")
        ok = [n for n in table["Algorithm"] if n in outs]
    with st.container(border=True):
        card_title("المقارنة", "Silhouette: كل ما قرّب لـ 1 أحسن · Davies-Bouldin: كل ما قل أحسن." if task == "clustering" else "")
        show_df(table.style.format(precision=4, na_rep="-"), hide_index=True)
    st.write("")
    with st.container(border=True):
        pick = w_select("res_pick_u", "الخوارزمية", ok)
        o = outs[pick]
        data = res.get("data")
        if task == "dimred":
            emb = o["emb"]
            c1, c2 = st.columns([3, 2])
            with c1:
                show_fig(plot_scatter2d(emb[:, :2] if emb.shape[1] > 1 else np.c_[emb[:, 0], np.zeros(len(emb))],
                                        title=f"{pick} — أول مكونين"))
            with c2:
                proj = pd.DataFrame(emb, columns=[f"comp_{i + 1}" for i in range(emb.shape[1])])
                show_df(proj.head(15).round(3), height=300)
            st.download_button("تحميل الناتج (.csv)", proj.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pick}_projection.csv", mime="text/csv")
            return
        labels = o
        proj = res["proj"]
        anomaly = task == "anomaly"
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plot_scatter2d(proj, labels, f"{pick} (PCA 2D)", anomaly=anomaly))
        with c2:
            if anomaly:
                flag = labels == -1
                st.caption(f"الشواذ: {int(flag.sum())} من {len(flag)}")
                show_df(data[flag].head(50), height=300)
            else:
                sizes = pd.Series(labels).value_counts().sort_index().rename("عدد الصفوف").to_frame()
                sizes.index = [("noise" if i == -1 else f"cluster {i}") for i in sizes.index]
                show_df(sizes, height=200)
        out = data.copy()
        if anomaly:
            out["is_anomaly"] = labels == -1
        else:
            out["cluster"] = labels
            prof = out.groupby("cluster").mean(numeric_only=True).round(2)
            with st.expander("بروفايل كل Cluster (متوسط الأعمدة الرقمية)", expanded=True):
                show_df(prof)
        st.download_button("تحميل الداتا + النتيجة (.csv)", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{pick}_labeled.csv", mime="text/csv")
    if task == "clustering":
        st.write("")
        with st.expander("كام Cluster مناسب؟ (KMeans)"):
            kmax = w_slider("kmax", "أقصى K", 3, 15, 10)
            if st.button("احسب"):
                df, _ = get_clean()
                with st.spinner("بيحسب…"):
                    S["kcurve"] = silhouette_by_k(cap_rows(df, int(S.get("max_rows", 50000))), res["features"],
                                                  res["config"], kmax)
            if S.get("kcurve"):
                ks, sil, inertia = S["kcurve"]
                a, b = st.columns(2)
                with a:
                    show_fig(plot_line(ks, sil, "Silhouette (الأعلى أحسن)", PALETTE[2], "silhouette"))
                with b:
                    show_fig(plot_line(ks, inertia, "Elbow — Inertia", PALETTE[1], "inertia"))
                st.caption(f"أعلى Silhouette عند K = {ks[int(np.argmax(sil))]}")


def page_res():
    page_header("res")
    res = S.get("res")
    if not res:
        empty_state("لسه مفيش نتائج", "درّب الخوارزميات الأول وهتلاقي المقارنة هنا.", "algo", "روح للخوارزميات")
        return
    st.markdown(f"<span class='badge'>{TASKS[res['task']]['label']}</span>", unsafe_allow_html=True)
    st.write("")
    (results_supervised if is_supervised(res["task"]) else results_unsupervised)(res)
    nav_row("algo", "code", "الكود الناتج")


# ==================== 7) الكود ====================
def page_code():
    page_header("code")
    try:
        ctx = code_context()
        if not ctx["chosen"]:
            raise ValueError("اختار خوارزمية واحدة على الأقل من صفحة الخوارزميات.")
        if not ctx["features"]:
            raise ValueError("اختار Features من صفحة التجهيز.")
        code = generate_code(ctx)
    except Exception as e:
        empty_state("مقدرتش أولّد الكود", str(e), "algo", "روح للخوارزميات")
        return
    with st.container(border=True):
        card_title("الكود الجاهز", "الأقسام: تحميل ← تنضيف و Outliers ← رسومات ← تجهيز ← موديلات وتقييم. بيتحدّث تلقائيًا مع أي تغيير.")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("تحميل .py", code.encode("utf-8"), file_name="ml_pipeline.py", mime="text/x-python")
        c2.download_button("تحميل Notebook", code_to_notebook(code), file_name="ml_pipeline.ipynb",
                           mime="application/x-ipynb+json")
        if ctx["source"].startswith("file:"):
            st.caption(f"غيّر مسار الملف «{ctx['source'][5:]}» في الكود لمكانه عندك. المكتبات: pandas, scikit-learn, matplotlib.")
        st.code(code, language="python")
    nav_row("res", None)


# ==================== سجل الموديلات المحفوظة (Sidebar) ====================
def saved_models_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 سجل الموديلات المحفوظة**")
    meta_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith("_meta.json"))
    if not meta_files:
        st.sidebar.info("لا توجد موديلات محفوظة بعد.")
        return
    names = [f[: -len("_meta.json")] for f in meta_files]
    selected = st.sidebar.selectbox("اختر موديل لمعاينة أدائه:", names, key="saved_model_pick")
    meta_path = os.path.join(MODELS_DIR, f"{selected}_meta.json")
    model_path = os.path.join(MODELS_DIR, f"{selected}.pkl")
    if st.sidebar.button("عرض تقرير الموديل"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.markdown(f"### 📊 تقرير الموديل المحفوظ: {selected}")
        st.write(f"**الخوارزمية:** {meta.get('algorithm', '-')} · **نوع المهمة:** {meta.get('task')} · "
                 f"**المقياس الأساسي:** {meta.get('main_metric', '-')}")
        st.write("**الأعمدة المستخدمة (Features):**")
        st.json(meta.get("features"))
        if meta.get("metrics"):
            st.write("**المقاييس:**")
            show_df(pd.DataFrame([meta["metrics"]]).T.rename(columns={0: "القيمة"}).astype(str))
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            st.sidebar.download_button("📥 تنزيل الموديل (.pkl)", f.read(), file_name=f"{selected}.pkl",
                                       mime="application/octet-stream")


# ==================== MAIN ====================
PAGE_FUNCS = {"data": page_data, "eda": page_eda, "clean": page_clean, "prep": page_prep,
              "algo": page_algo, "res": page_res, "code": page_code}

sidebar()
_page = S.get("page", "data")
if _page != "data" and S.get("raw") is None:
    empty_state("ابدأ بالداتا", "ارفع ملف أو اختار داتا تجريبية الأول.", "data", "روح لصفحة البيانات")
else:
    PAGE_FUNCS[_page]()
saved_models_sidebar()
