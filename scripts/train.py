# LightGBM GBDT with KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
import gc
import os
import pickle
import pandas as pd

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from scripts.helper_functions import display_importances



# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds=10, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]  # Target yok
    test_df = df[df['TARGET'].isnull()]    # Target var
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    # K-fold veri kümesini k katlara bölen çapraz doğrulayıcıdır.(KFold is a cross-validator that divides the dataset into k folds)
    # dengesiz veri kümesinde her veriden aynı oranda k kat aldığı için doğru sonuç verme oranı SKfold a göre düşüktür.
    # Eğer dengesiz veri problemi ile uğraşıyorsak SKfold kullanımı daha yerinde olur.
    # (Therefore, the answer to this question is we should prefer StratifiedKFold over KFold when dealing with classifications tasks with imbalanced class distributions.)
    # StratifiedKfold her veri kümesini kendi oranında k katlara bölen çapraz doğrulayıcıdır.
    # (Stratified is to ensure that each fold of dataset has the same proportion of observations with a given label.)
    # örn: A değişkeninden 12 veri, B değişkeninden 4 veri gelsin. k=4 olsun (aralarında 1/3 oran var)
    # O halde bu orana bakarak A dan 3 veri noktası , B den 1 veri noktası getirecektir.


    # n_splits=
    # shuffle= Default değeri False. Verilerin gruplara ayrılmadan önce karıştırılmaması ön koşul. Biz burada karıştırdık.
    # random_state=
    # shuffle False olduğunda random_state kullanılmaz.

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])  # validasyon için oluşturuldu
    sub_preds = np.zeros(test_df.shape[0])   # submission için oluşturuldu
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4, # ?????
            n_estimators=10000, # ağaç sayısı
            learning_rate=0.02, # ağaç ölçeklendirme. Değer küçükse tahmin başarısı yükselir, öğrenim süresi artar, overfit ihtimali artar.
            # num_leaves = 2^(max_depth) ??
            num_leaves=34, # her iterasyonda oluşturulacak karar ağacının yaprak sayısı. Sayının yüksek olması performansı düşürür.
            colsample_bytree=0.9497036, # ağaç oluştururken değişkenlerin alt örnek oranı
            subsample=0.8715623, # Eğitim verisinin alt örnek oranı
            max_depth=8,  # ağaç derinliği. Çok dallanma overfit, az dallanma eksik öğrenmedir.
            reg_alpha=0.041545473, # L1 ağırlığını düzenleme
            reg_lambda=0.0735294,  # L2 ağırlığını düzenleme
            min_split_gain=0.0222415, # ağacın bir yaprak düğümünde daha fazla bölme yapmak için kaybı min indirme
            min_child_weight=39.3259775, # bir yaprakta ihtiyaç duyulan minimum ağırlık
            silent=-1, # işlem anında mesaj yazılıp, yazılmaması
            verbose=-1, ) #

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], # validasyona alınacak kümeler seçildi
                eval_metric='auc', verbose=200, early_stopping_rounds=200)  # sınıflandırma prblmei olduğu için metrik seçimi auc

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv("outputs/predictions/reference_submission.csv", index=False)
    display_importances(feature_importance_df)
    return feature_importance_df
