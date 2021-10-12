import copy

import eli5
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

dataset = pd.read_excel("data.xlsx")
dataset.dropna(inplace=True)
cleanup_nums = {1: True, 0: False}
dataset["event"].replace(cleanup_nums, inplace=True)

# Building training and testing sets #
index_train, index_test = train_test_split(range(dataset.shape[0]), test_size=0.3)
data_train = dataset.iloc[index_train, :]
data_test = dataset.iloc[index_test, :]

y_train, y_test = Surv.from_arrays(data_train.iloc[:, -1], data_train.iloc[:, -2]), Surv.from_arrays(data_test.iloc[:, -1], data_test.iloc[:, -2])
X_train, X_test = data_train.iloc[:, range(data_train.shape[1] - 2)], data_test.iloc[:, range(data_test.shape[1] - 2)]

rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1)
rsf.fit(X_train, y_train)
rsf.score(X_test, y_test)

surv = rsf.predict_survival_function(X_test)

# for i, s in enumerate(surv):
#     plt.step(rsf.event_times_, s, where="post", label=str(i))
# plt.ylabel("Survival probability")
# plt.xlabel("Time in days")
# plt.grid(True)
# plt.legend()
#
#
# surv = rsf.predict_cumulative_hazard_function(X_test)
# for i, s in enumerate(surv):
#     plt.step(rsf.event_times_, s, where="post", label=str(i))
# plt.ylabel("Cumulative hazard")
# plt.xlabel("Time in days")
# plt.grid(True)
# plt.legend()

perm = PermutationImportance(rsf, n_iter=15)
perm.fit(X_test, y_test)
pd.DataFrame({"name":dataset.columns[range(dataset.shape[1]-2)],"importance":perm.feature_importances_}).to_excel("importance.xlsx")
eli5.show_weights(perm, feature_names=dataset.columns[range(dataset.shape[1]-2)])