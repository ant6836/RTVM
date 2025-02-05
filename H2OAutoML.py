# %%
import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt


# %%
def get_ece(model, test, threshold):
    df_pred = model.predict(test).as_data_frame()
    df_real = test.as_data_frame()
    df_pred["real"] = df_real["life_real"]

    threshold = 0.5
    for i in df_pred.index:
        if df_pred["Y"][i] > threshold:
            df_pred["predict"][i] = "Y"
        else:
            df_pred["predict"][i] = "N"

    bin_dict = {
        "b1": {"prob": [], "match_cnt": 0},
        "b2": {"prob": [], "match_cnt": 0},
        "b3": {"prob": [], "match_cnt": 0},
        "b4": {"prob": [], "match_cnt": 0},
        "b5": {"prob": [], "match_cnt": 0},
    }

    for i in df_pred.index:
        prob = df_pred["Y"][i]
        pred_label = df_pred["predict"][i]
        real_label = df_pred["real"][i]

        if prob <= 0.2:
            bin_dict["b1"]["prob"].append(prob)
            if pred_label == real_label:
                bin_dict["b1"]["match_cnt"] += 1
        elif prob <= 0.4:
            bin_dict["b2"]["prob"].append(prob)
            if pred_label == real_label:
                bin_dict["b2"]["match_cnt"] += 1
        elif prob <= 0.6:
            bin_dict["b3"]["prob"].append(prob)
            if pred_label == real_label:
                bin_dict["b3"]["match_cnt"] += 1
        elif prob <= 0.8:
            bin_dict["b4"]["prob"].append(prob)
            if pred_label == real_label:
                bin_dict["b4"]["match_cnt"] += 1
        else:
            bin_dict["b5"]["prob"].append(prob)
            if pred_label == real_label:
                bin_dict["b5"]["match_cnt"] += 1

    n = df_pred.shape[0]
    ece = 0
    for key in bin_dict.keys():
        bin_prob_list = bin_dict[key]["prob"]
        bin_match_cnt = bin_dict[key]["match_cnt"]

        conf = sum(bin_prob_list) / len(bin_prob_list)
        acc = bin_match_cnt / len(bin_prob_list)

        bin_ece = len(bin_prob_list) / n * abs(conf - acc)

        ece += bin_ece

    return ece


# %%
def get_performance(model, test):
    perf = model.model_performance(test)
    threshold = 0.5

    accuracy = perf.accuracy([threshold])[0][1]
    precision = perf.precision([threshold])[0][1]
    recall = perf.recall([threshold])[0][1]
    f1 = perf.F1([threshold])[0][1]
    yj = perf.sensitivity([threshold])[0][1] + perf.specificity([threshold])[0][1] - 1
    mcc = perf.mcc([threshold])[0][1]
    ece = get_ece(model, test, threshold)

    perf_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yj": yj,
        "mcc": mcc,
        "ece": ece,
    }

    return perf_dict


# %%
def get_confusion_matrix(model, test):
    preds = model.predict(test)
    df_preds = preds.as_data_frame()

    threshold = 0.5
    for i in df_preds.index:
        if df_preds["Y"][i] > threshold:
            df_preds["predict"][i] = "Y"
        else:
            df_preds["predict"][i] = "N"

    df_real["pred"] = df_preds["predict"]

    confusion_matrix = pd.crosstab(
        df_real["life_real"], df_real["pred"], rownames=["Real"], colnames=["Pred"]
    )

    return confusion_matrix


# %%
def get_cv_result(model):
    cv_result = model.cross_validation_predictions()
    cv_result_list = list()

    for cv_i in cv_result:
        df_cv = cv_i.as_data_frame()
        df_cv["real"] = h2o_train.as_data_frame()["life_real"]
        df_cv = df_cv[df_cv["N"] != 0]
        cv_result_list.append(df_cv)

    return cv_result_list


# %%
def show_reliability_diagram(cv_result_list, model_name, legend=True):
    plt.rcParams["font.family"] = "Times New Roman"

    plt.figure(figsize=(6, 6))

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, zorder=10)

    for v in [0.2, 0.4, 0.6, 0.8]:
        plt.axvline(v, 0, 1, color="lightgray", linestyle="--", linewidth=1)
        plt.axhline(v, 0, 1, color="lightgray", linestyle="--", linewidth=1)

    prob_pred_all = [[] for _ in range(5)]
    prob_true_all = [[] for _ in range(5)]

    for i, df_for_curve in enumerate(cv_result_list):
        prob_true, prob_pred = calibration_curve(
            y_true=df_for_curve["real"],
            y_prob=df_for_curve["Y"],
            pos_label="Y",
            n_bins=5,
        )
        for j in range(5):
            prob_pred_all[j].append(prob_pred[j])
            prob_true_all[j].append(prob_true[j])
        plt.plot(
            prob_pred,
            prob_true,
            label=f"CV {i+1}",
            linestyle="-",
            linewidth=0.5,
            zorder=10,
        )

    prob_pred_avg = [np.mean(i) for i in prob_pred_all]
    prob_true_avg = [np.mean(i) for i in prob_true_all]

    prob_true_std = [np.std(i) for i in prob_true_all]

    prob_true_upper = [prob_true_avg[i] + prob_true_std[i] / 2 for i in range(5)]
    prob_true_lower = [prob_true_avg[i] - prob_true_std[i] / 2 for i in range(5)]

    plt.fill_between(
        prob_pred_avg,
        prob_true_upper,
        prob_true_lower,
        color="silver",
        alpha=0.8,
        label="STD",
        zorder=5,
    )

    plt.plot(
        prob_pred_avg,
        prob_true_avg,
        label="AVG",
        linestyle="-",
        linewidth=2.5,
        color="black",
        zorder=10,
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Predicted probability", fontsize=15)
    plt.ylabel("Fraction of positives", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.savefig(f"./diagram_{model_name}.png")

    plt.show()


# %%
h2o.init()
h2o.no_progress()

# %%
data = pd.read_csv("./training_set420.csv", encoding="utf-8")
data = data.drop("lifetime", axis=1)

# %%
train, test = train_test_split(
    data, test_size=0.3, stratify=data["life_real"], random_state=1
)

# %%
X_train = train.drop("life_real", axis=1)
y_train = train["life_real"]

tl = TomekLinks(sampling_strategy="majority")
X_train_under, y_train_under = tl.fit_resample(X_train, y_train)

# %%
X_test = test.drop("life_real", axis=1)
y_test = test["life_real"]

# %%
y = "life_real"
x = list(data.columns)

x.remove(y)

# %%
train_for_h2o = pd.concat([y_train_under, X_train_under], axis=1)
test_for_h2o = pd.concat([y_test, X_test], axis=1)

# %%
train_for_h2o = pd.read_pickle("./train_for_h2o.pkl")
test_for_h2o = pd.read_pickle("./test_for_h2o.pkl")

# %%
h2o_train = h2o.H2OFrame(train_for_h2o)
h2o_test = h2o.H2OFrame(test_for_h2o)

# %%
h2o_train[y] = h2o_train[y].asfactor()
h2o_test[y] = h2o_test[y].asfactor()

# %%
max_runtime_secs = 3600

aml = H2OAutoML(
    max_runtime_secs=max_runtime_secs,
    nfolds=10,
    include_algos=["DRF", "GLM", "DeepLearning", "XGBoost"],
    seed=1,
    stopping_metric="logloss",
    stopping_rounds=5,
    keep_cross_validation_predictions=True,
)
aml.train(x=x, y=y, training_frame=h2o_train)

# %%
lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
df_lb = lb.as_data_frame()

# %%
algo_name = "NN"
df_lb.to_csv(
    f"./AutoML/JJBR/leaderboard_{algo_name}.csv", encoding="utf-8", index=False
)

# %%
for i in df_lb.index:
    if i == 0:
        continue
    if i == 10:
        break
    model_id = df_lb["model_id"][i]
    m = h2o.get_model(model_id)
    m_path = h2o.save_model(model=m, path=f"./AutoML/JJBR/{algo_name}", force=True)


# %%
best_RF = aml.get_best_model(algorithm="DRF")
best_RF_path = h2o.save_model(model=best_RF, path="./RF", force=True)

best_LR = aml.get_best_model(algorithm="GLM")
best_LR_path = h2o.save_model(model=best_LR, path="./LR", force=True)

best_NN = aml.get_best_model(algorithm="DeepLearning")
best_NN_path = h2o.save_model(model=best_NN, path="./NN", force=True)

best_XG = aml.get_best_model(algorithm="XGBoost")
best_XG_path = h2o.save_model(model=best_XG, path="./XG", force=True)

# %%
perf_RF = get_performance(best_RF, h2o_test)
perf_LR = get_performance(best_LR, h2o_test)
perf_NN = get_performance(best_NN, h2o_test)
perf_XG = get_performance(best_XG, h2o_test)

# %%
df_real = h2o_test.as_data_frame()

# %%
confusion_matrix_RF = get_confusion_matrix(best_RF, h2o_test)
confusion_matrix_LR = get_confusion_matrix(best_LR, h2o_test)
confusion_matrix_NN = get_confusion_matrix(best_NN, h2o_test)
confusion_matrix_XG = get_confusion_matrix(best_XG, h2o_test)

# %%
cv_result_RF = get_cv_result(best_RF)
cv_result_LR = get_cv_result(best_LR)
cv_result_NN = get_cv_result(best_NN)
cv_result_XG = get_cv_result(best_XG)

# %%
show_reliability_diagram(cv_result_RF, model_name="RF", legend=True)

# %%
show_reliability_diagram(cv_result_LR, model_name="LR", legend=True)

# %%
show_reliability_diagram(cv_result_NN, model_name="NN", legend=True)

# %%
show_reliability_diagram(cv_result_XG, model_name="XG", legend=True)
