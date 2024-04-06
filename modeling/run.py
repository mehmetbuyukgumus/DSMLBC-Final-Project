import sys
sys.path.append("modeling")
import modelingWithRegrossor as mr
import modelingWithClassifiers as mc

def main(regressor = True):
    if regressor:
        df = mr.load_data()
        df = mr.drop_na(df)
        float_cols , obj_cols, int_cols = mr.grab_col_names(df)
        df = mr.standarditaziton(df, float_cols)
        df = mr.encoding(df, "type")
        mr.base_models(df, cv=5)
    else:
        df = mc.prep_data()
        df = mc.encoding(df, "type")
        df = mc.dropna(df)
        df = mc.Standardization(df)
        mc.base_models(df, cv=5)

if __name__ == "__main__":
    main(regressor=False)
    
    
    
# Regresyon Sonuçları
# ----------------------------------------------
# neg_mean_squared_error: -0.5549 (LR)
# neg_mean_squared_error: -0.5549 (Ridge)
# neg_mean_squared_error: -0.5935 (XGBM)
# neg_mean_squared_error: -0.5264 (LGBM)
# neg_mean_squared_error: -0.7698 (ElasticNey)
# neg_mean_absolute_error: -0.5789 (LR)
# neg_mean_absolute_error: -0.5789 (Ridge)
# neg_mean_absolute_error: -0.5989 (XGBM)
# neg_mean_absolute_error: -0.5639 (LGBM)
# neg_mean_absolute_error: -0.6884 (ElasticNey)
# r2: 0.2604 (LR)
# r2: 0.2604 (Ridge)
# r2: 0.2047 (XGBM)
# r2: 0.2988 (LGBM)
# r2: -0.0271 (ElasticNey)

# Sınıflandırma Sonuçları
# ----------------------------------------------
# f1: 0.7261 (LR)
# f1: 0.6569 (RF)
# f1: 0.6943 (GB)
# f1: 0.7155 (ADA)
# f1: 0.6951 (XGBM)
# f1: 0.6934 (LGBM)
# accuracy: 0.6893 (LR)
# accuracy: 0.6704 (RF)
# accuracy: 0.685 (GB)
# accuracy: 0.6879 (ADA)
# accuracy: 0.6758 (XGBM)
# accuracy: 0.6825 (LGBM)
# roc_auc: 0.791 (LR)
# roc_auc: 0.7826 (RF)
# roc_auc: 0.7919 (GB)
# roc_auc: 0.772 (ADA)
# roc_auc: 0.7747 (XGBM)
# roc_auc: 0.7848 (LGBM)
# precision: 0.7896 (LR)
# precision: 0.7978 (RF)
# precision: 0.7937 (GB)
# precision: 0.7906 (ADA)
# precision: 0.7926 (XGBM)
# precision: 0.7951 (LGBM)

