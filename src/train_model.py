import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from IPython import embed
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import catboost
import random


def exec_cross_validation(args, X_train, y_train, X_valid, y_valid,
                                X_test, y_test):
    if args['algo_name'] == 'LightGBM':
        i = 0
        nfold = args['n_folds']
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
        for train, eval in skf.split(X_train, y_train):
            X_train_skf = X_train.iloc[train]
            y_train_skf = y_train.iloc[train]
            params = set_LGBM_fixed_params()
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_skf, y_train_skf,
                    eval_set=(X_valid, y_valid),
                    verbose=-1,
                    early_stopping_rounds=100)
            predict_proba_nfold = model.predict_proba(X_test,
                                                num_iteration=model.best_iteration_)
            if i == 0:
                predict_probas = np.array(predict_proba_nfold)
            else:
                predict_probas += predict_proba_nfold
            i += 1
        predict_proba = predict_probas / nfold

    elif args['algo_name'] == 'CatBoost':
        i = 0
        nfold = args['n_folds']
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
        for train, eval in skf.split(X_train, y_train):
            X_train_skf = X_train.iloc[train]
            y_train_skf = y_train.iloc[train]
            model = catboost.CatBoostClassifier(iterations=2000,
                                              use_best_model=True,
                                              loss_function='MultiClass',
                                              eval_metric='TotalF1',
                                              classes_count=3)
            model.fit(X_train_skf, y_train_skf,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=200)
            predict_proba_nfold = model.predict_proba(X_test)
            if i == 0:
                predict_probas = np.array(predict_proba_nfold)
            else:
                predict_probas += predict_proba_nfold
            i += 1
        predict_proba = predict_probas / nfold

    try:
        test_logloss = log_loss(y_test, predict_proba)
        predict = [
            np.argmax(proba) if max(proba) >= 0.4 else 0
            for proba in predict_proba
        ]
        mat = confusion_matrix(y_test, predict)
        score = (mat[1,1] + mat[2,2]) / \
                (mat[1,1] + mat[2,2] + mat[1,2] + mat[2,1] + mat[0,1]/5 + mat[0,2]/5 + 1e-8)

        print('test_logloss: {}'.format(test_logloss))
        print('f1_score: {}'.format(score))
        print('mat: {}'.format(mat))

        n_nonzero_predict = len([p for p in predict if p in [1, 2]])

        for i in range(len(predict)):
            if n_nonzero_predict > 20:
                if random.random() > 20 / n_nonzero_predict:
                    continue
            if predict[i] in [1, 2]:
                if y_test[i] == predict[i]:
                    print('  TP or TN @ {}'.format(str(y_test.index[i])))
                elif y_test[i] == 0:
                    print('  Neutral @ {}'.format(str(y_test.index[i])))
                else:
                    print('  FP or FN @ {}'.format(str(y_test.index[i])))

        return -1 * score, test_logloss, model
    except:
        return 1000, 1000, None


def plot_param_importance(model, type):
    if type == 'original':
        import matplotlib.pyplot as plt
        lgb.plot_importance(model, figsize=(30,12))
        plt.show()
        df = pd.DataFrame(model.feature_importances_, index=X_train_skf.columns)
        list(df[df>40].dropna().sort_values(by=[0]).index)
        list(df[df<40].dropna().sort_values(by=[0]).index)
    elif type == 'shap': # shap
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_skf)
        shap.summary_plot(shap_values, X_train_skf, plot_type="bar")
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_train_skf.iloc[0,:])


def set_LGBM_fixed_params():
   return {
        'objective': 'multiclass',
        'class_weight': 'balanced',
        'num_boost_round': 500,
        'learning_rate': 0.02,
        'num_class': 3,
        'silent': True
    }
