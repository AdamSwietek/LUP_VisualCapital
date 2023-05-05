import pandas as pd
import numpy as np
import geopandas as gpd



def dat_split(input_dat, id_gebs):
    X     = input_dat.loc[input_dat.ID_Geb.isin(id_gebs),:]
    X_rem = input_dat.loc[~input_dat.ID_Geb.isin(id_gebs),:]
    return X, X_rem

from sklearn.model_selection import train_test_split

def train_test_split_by_commune(input_df,sel_feats, target_var,rseed, test_size = .3):
    gmd_train, gmd_test = train_test_split(input_df.GMDNAME.unique(), test_size=test_size, random_state=rseed)
    #     dat.query('GMDNAME in @gmd_train').shape, dat.query('GMDNAME in @gmd_test')
    train_gebs = input_df.query('GMDNAME in @gmd_train')['ID_Geb'].unique()
    test_gebs  = input_df.query('GMDNAME in @gmd_test')['ID_Geb'].unique()

    X_train = pd.DataFrame(input_df.query('GMDNAME in @gmd_train')).loc[:,sel_feats]#.values
    y_train = pd.Series(input_df.query('GMDNAME in @gmd_train')[target_var])#.values[:,None]

    X_test = pd.DataFrame(input_df.query('GMDNAME in @gmd_test')).loc[:,sel_feats]#.values
    y_test = pd.Series(input_df.query('GMDNAME in @gmd_test')[target_var])#.values[:,None]

    return {'train':(X_train,y_train,train_gebs),'test':(X_test,y_test,test_gebs)}

from sklearn.metrics import r2_score

def evaluate_model(model, input_data, select_feats, target_var):

    input_data['pred'] = model.predict(input_data.loc[:,select_feats])
    rsquare_building = r2_score(input_data[target_var].values, input_data['pred'].values )

    commune_pred = input_data.groupby('GMDNAME').apply(
        lambda x: pd.Series({
                'mean_pred': x.pred.mean(), 
                'median_pred': x.pred.median(), 
                'sd_pred': x.pred.std(), 
                'min_pred': x.pred.min(),
                'max_pred': x.pred.max(),
                'q3_pred' : x.pred.quantile(.75),
                target_var: x[target_var].mean()
                }))
    
    rsquare_commune = r2_score(commune_pred[target_var].values,commune_pred['mean_pred'] )

    return dict(model = model,
                commune_pred = commune_pred,
                rsquare_building = rsquare_building,   
                rsquare_commune = rsquare_commune )