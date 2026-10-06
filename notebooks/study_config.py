"""Shared study configuration for all three arms.

Every setting that the proposal requires to be identical across arms (features,
preprocessing, classifiers, hyperparameter grids, cross validation settings and
the classification threshold) is defined once here and imported by
global_model_armA.ipynb, clinical_stratified_armB.ipynb and
data_driven_armC.ipynb. Keeping a single definition means the arms cannot drift
apart when one notebook is edited, so any difference between arms comes from
the stratification strategy alone.
"""

import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Reproducibility and cross validation
RANDOM_STATE = 2
K_OUTER = 10
K_INNER = 10

# Predicted probability at or above this value is classed as disease present.
# 0.5 assigns each patient to the more likely class.
THRESHOLD = 0.5

# Tuning criterion used by every GridSearchCV in every arm.
TUNING_SCORING = "roc_auc"

# All 13 clinical features named in the proposal (Section 3.1).
FEATURE_SET = "all"

FEATURE_GROUPS = {
    "all": {
        "continuous": ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        "nominal": ["cp", "restecg", "slope", "thal"],
        "binary": ["sex", "fbs", "exang"],
    },
    # Earlier 9 feature subset. It was chosen from a ranking computed on all
    # 297 patients, so using it inside cross validation leaks information from
    # the validation folds. Kept only so that earlier results can be reproduced.
    "selected": {
        "continuous": ["age", "thalach", "oldpeak", "ca"],
        "nominal": ["cp", "slope", "thal"],
        "binary": ["sex", "exang"],
    },
}

# Fixed clinical coding of each nominal feature (from the UCI documentation,
# not estimated from data), so every fold's encoder knows every valid code even
# when a small training subset happens not to contain it.
NOMINAL_CATEGORIES = {
    "cp": [1.0, 2.0, 3.0, 4.0],
    "restecg": [0.0, 1.0, 2.0],
    "slope": [1.0, 2.0, 3.0],
    "thal": [3.0, 6.0, 7.0],
}

# Full classifier names, used as the model label in every saved result file.
LOGREG = "Logistic regression"
RF = "Random forest"
SVM = "Support vector machine"
MODEL_NAMES = [LOGREG, RF, SVM]

# Paths (notebooks live in notebooks/, data and outputs one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "heart+disease")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
DIAGNOSTICS_DIR = os.path.join(PROJECT_ROOT, "diagnostics")
FOLD_FILE = os.path.join(PROJECT_ROOT, "fold_id.csv")
MANIFEST_FILE = os.path.join(PROJECT_ROOT, "run_manifest.json")
CLEAN_DATA_FILE = os.path.join(DATA_DIR, "cleveland_clean.csv")


def get_feature_groups(feature_set=FEATURE_SET, exclude=()):
    """Copy of the continuous / nominal / binary lists for feature_set, with
    any column in `exclude` removed (Arm B removes `sex`)."""
    return {
        group: [c for c in cols if c not in exclude]
        for group, cols in FEATURE_GROUPS[feature_set].items()
    }


def get_feature_list(feature_set=FEATURE_SET, exclude=()):
    """Flat list of raw feature columns for feature_set."""
    g = get_feature_groups(feature_set, exclude)
    return g["continuous"] + g["nominal"] + g["binary"]


def get_preprocessor(feature_set=FEATURE_SET, exclude=()):
    """Encoding followed by standardisation of every resulting column.

    1. Nominal features are one hot encoded with the first level dropped (the
       dropped level becomes the reference category), continuous and binary
       features pass through unchanged.
    2. Every column of the encoded matrix is then standardised to mean 0 and
       variance 1. Penalised logistic regression and the SVM both depend on
       the scale of their inputs: an L1 or L2 penalty shrinks every
       coefficient by the same amount, which is only fair if every predictor
       is on the same scale. Standardising only the continuous features would
       leave 0/1 columns with a much smaller spread, so their coefficients
       would be penalised more heavily. Random forests are unaffected.

    The whole object is only ever fitted inside a Pipeline on training data,
    so the means and standard deviations never see validation patients.
    """
    g = get_feature_groups(feature_set, exclude)
    encoder = ColumnTransformer(
        transformers=[
            ("continuous", "passthrough", g["continuous"]),
            ("nominal", OneHotEncoder(categories=[NOMINAL_CATEGORIES[c] for c in g["nominal"]],
                                      drop="first", sparse_output=False), g["nominal"]),
            ("binary", "passthrough", g["binary"]),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("encode", encoder), ("scale", StandardScaler())])


def get_models(feature_set=FEATURE_SET, exclude=()):
    """The three classifiers and their hyperparameter grids, as
    {name: (pipeline, grid)}. Identical in every arm; only Arm B passes
    exclude=("sex",)."""

    def pipe(estimator):
        return Pipeline([("prep", get_preprocessor(feature_set, exclude)), ("clf", estimator)])

    return {
        LOGREG: (
            pipe(LogisticRegression(solver="liblinear", max_iter=5000, random_state=RANDOM_STATE)),
            {
                "clf__C": np.logspace(-3, 2, 10),
                "clf__l1_ratio": [0.0, 1.0],  # 0 = L2 (ridge), 1 = L1 (lasso)
            },
        ),
        RF: (
            pipe(RandomForestClassifier(random_state=RANDOM_STATE)),
            {
                "clf__max_depth": [2, 3, 4, 5],
                "clf__min_samples_leaf": [2, 4, 8],
                "clf__n_estimators": [50, 100],
            },
        ),
        SVM: (
            # Sigmoid (Platt) calibration gives the SVM predicted probabilities,
            # needed for ROC AUC and for the shared 0.5 threshold.
            pipe(CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), method="sigmoid", cv=5, ensemble=False)),
            {
                "clf__estimator__C": np.logspace(-3, 2, 6),
                "clf__estimator__kernel": ["linear", "rbf"],
                "clf__estimator__gamma": ["scale"],
            },
        ),
    }


def make_dirs():
    for d in (RESULTS_DIR, PREDICTIONS_DIR, DIAGNOSTICS_DIR):
        os.makedirs(d, exist_ok=True)
