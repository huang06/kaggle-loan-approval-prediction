#!/usr/bin/env python

# In[1]:


from __future__ import annotations

import datetime
import logging

import catboost
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from xgboost import XGBRegressor

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level="INFO")
log = logging.getLogger("notebook")


# In[2]:


train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
test_id_list = list(test_df["id"])

train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])


# In[3]:


train_df["person_age_1"] = (
    train_df["person_age"].clip(lower=0, upper=99).map(lambda x: x // 10).value_counts()
)
test_df["person_age_1"] = test_df["person_age"].clip(lower=0, upper=99).map(lambda x: x // 10).value_counts()


# In[4]:


cat_features = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
num_features = [
    'person_age',
    'person_age_1',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
]


# ## Hyper parameter tunning with Optuna

# In[ ]:


def objective(trial):
    cv = StratifiedKFold(5, shuffle=True, random_state=9999)
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "eval_metric": "AUC",
        "cat_features": cat_features,
        "random_state": 9999,
        "iterations": 1000,
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 1.0, log=True),
    }
    score_list = []
    for train_idx, val_idx in cv.split(train_df, y=train_df["loan_status"]):
        train_x = train_df.iloc[train_idx][cat_features + num_features]
        train_y = train_df.iloc[train_idx]["loan_status"]
        valid_x = train_df.iloc[val_idx][cat_features + num_features]
        valid_y = train_df.iloc[val_idx]["loan_status"]
        cb_clf = catboost.CatBoostClassifier(**param)
        cb_clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=200)
        valid_y_pred = cb_clf.predict_proba(valid_x)[:, 1]
        score = roc_auc_score(valid_y, valid_y_pred)
        score_list.append(score)
    return np.mean(score_list)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
cb_clf_params = study.best_params

print("Done")
