import numpy as np
import pandas as pd
import geopandas as gpd
import quantecon as qe
import re



############################################
# SETUP
############################################

def breakoutCols(input_df, txt = 'Sh'):
    import re
    sh_cols = input_df.columns[input_df.columns.str.contains(txt)].tolist()
    split_col_names = [re.split(r'(\d+)',col_nx) for col_nx in sh_cols]

    sh_df = pd.DataFrame(split_col_names, 
                            index =sh_cols, 
                            columns=['Distance','dist_id','Obstruction','obstr_id','POI','poi_id','drop_me'] )

    sh_df['Distance'] = sh_df['Distance'] + sh_df['dist_id']
    sh_df['Obstruction'] = sh_df['Obstruction'] + sh_df['obstr_id']
    sh_df['POI'] = sh_df['POI'] + sh_df['poi_id']
    sh_df = sh_df.drop(columns=['dist_id','obstr_id','poi_id','drop_me'])

    return sh_df

def getTable(func, data_lst, col_names, col_labels, prefix_nm):

    output_lst = []
    for data in data_lst:
        output_lst.append(pd.concat([func(data[col_names[ix]]) for ix in range(len(col_names))], 
                                    axis=1, 
                                    keys = col_labels))

    output = [opt.add_prefix(prefix_nm) for opt in output_lst]
    return output

############################################
# INDIV. FUNCS
############################################

def calcClosest(x):
    s = x.min(axis	=1)
    return s

def calcSummI(x):
    s = x.sum(axis = 1)
    return s 
def calcMeanI(x):
	s = x.mean(axis = 1)
	return s
def calcRichness(x, threshold = 0):
    richness = (x>threshold).sum(axis = 1)
    return richness
def gini_coefficient(y):
    r"""
    Implements the Gini inequality index

    Parameters
    ----------
    y : array_like(float)
        Array of income/wealth for each individual.
        Ordered or unordered is fine

    Returns
    -------
    Gini index: float
        The gini index describing the inequality of the array of income/wealth

    References
    ----------

    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    n = len(y)
    i_sum = np.zeros(n)
    for i in range(n):
        for j in range(n):
            i_sum[i] += abs(y[i] - y[j])
    return np.sum(i_sum) / (2 * n * np.sum(y))

def calcShanonWeaver(x):
    # feed proportion % as decimals
    x = x.div(x.sum(1), 0)
    s = x*np.log(x)
    s[np.isnan(s)] = 0

    # s = s.replace(np.nan,0)   
    # return -sum(s)
    return -np.sum(s,axis=1)

############################################
# AGGREGATOR FUNCS
############################################

def get_maxvsh_scores(chunk):

    sh_col_df = breakoutCols(chunk, txt = 'Sh')
    sh_cols = list(sh_col_df.index)

    view_elements = sh_col_df.fillna('sky').groupby('Obstruction').apply(lambda x: list(x.index))
    
    summi_unobs_ = getTable(func = calcSummI, 
                                data_lst = [chunk[sh_cols]], 
                                col_names = list(view_elements.values), 
                                col_labels = list(view_elements.index), 
                                prefix_nm = 'maxvsh_')[0]
    chunk_view = pd.concat([summi_unobs_, chunk.loc[:,['ID_Geb']]],axis=1)

    chunk_trx = chunk.loc[:,['ID_Geb','FassPktX', 'FassPktY', 'FassPktZ']]

    agg_view = chunk_view.groupby('ID_Geb').max()
    agg_trx = chunk_trx.groupby('ID_Geb').mean()
    agg_= pd.concat([agg_view,agg_trx], axis = 1)
    return {'score':agg_view, 'coords':agg_trx,'full':agg_}

def get_visual_access_score(chunk):
    #----------------------------------------------------------------------------
    #get visual access
    #----------------------------------------------------------------------------
    sh_col_df = breakoutCols(chunk, txt = 'Sh')
    sh_cols = list(sh_col_df.index)
    # sum_cols = chunk.columns[chunk.columns.str.contains('sum')]
    # other_cols =chunk.columns[~chunk.columns.isin(sh_cols.tolist()+ sum_cols.tolist())]
       
    view_elements = sh_col_df.fillna('sky').groupby('Obstruction').apply(lambda x: list(x.index))

    chunk = chunk.fillna(0)
    summi_unobs_ = getTable(func = calcSummI, 
                                    data_lst = [chunk[sh_cols]], 
                                    col_names = list(view_elements.values), 
                                    col_labels = list(view_elements.index), 
                                    prefix_nm = 'vwa_')[0]
    chunk_view = pd.concat([summi_unobs_>0.01, chunk.loc[:,['ID_Geb']]],axis=1)
    
    # chunk_trx = chunk.loc[:,other_cols]

    agg_visual_access = chunk_view.groupby('ID_Geb').mean()
    # agg_trx = chunk_trx.groupby('ID_Geb').mean()
    # agg_= pd.concat([agg_visual_access,agg_trx], axis = 1)
    return {'score':agg_visual_access}

def get_meanvsh_scores(chunk):

    sh_col_df = breakoutCols(chunk, txt = 'Sh')
    
    sh_cols = chunk.columns[chunk.columns.str.contains('Sh')]
    sum_cols = chunk.columns[chunk.columns.str.contains('sum')]
    other_cols =['FassPktX', 'FassPktY', 'FassPktZ']
    other_cols2 = ['Stockwerk']
        
    chunk = chunk.fillna(0)
    agg_view = chunk.groupby('ID_Geb')[sh_cols].mean()
    agg_other = chunk.groupby('ID_Geb')[other_cols].mean()
    agg_other2 = chunk.groupby('ID_Geb')[other_cols2].max()
    
    agg_= pd.concat([agg_view,agg_other,agg_other2], axis = 1)
    
    return {'score':agg_view, 'coords':agg_other,'topflr':agg_other2, 'full':agg_}

def get_sentinment_score(data):

    data = get_meanvsh_scores(data)['score']
    # LABELS/SENTIMENT SCORES-----------------------------------------------------------------------------
    sentiment_map = {'Abb7': 'Neg','Abw14':'Neg','Flu18':'Neg','Geb12':0,'Gew1':'Pos','Hel19':'Neg','Keh15':'Neg',
                    'Kue8':'Pos','Lan17':'Neg','Lan10':'Pos','Nat3':'Pos','Sak13':'Pos','Sie9':'Pos','Ueb5':'Neg','Ver6':'Neg',
                    'Ver11':'Neg','Was16':'Pos', 'sky':0, 'Dac1':0, 'Veg3':0, 'Fas2':0}
    sh_col_df = breakoutCols(data, txt = 'Sh')
    sh_cols = list(sh_col_df.index)
    sh_col_df['sentiment'] = [sentiment_map[d] for d in sh_col_df.fillna('sky')['Obstruction']]
    
    obs_condition = sh_col_df['POI'].isna()
    view_sentiment = sh_col_df.loc[(obs_condition),:].groupby('sentiment').apply(lambda x: list(x.index))

    summi_sentiment_ = getTable(func = calcSummI, 
                                data_lst = [data[sh_cols]], 
                                col_names  = list(view_sentiment.values), 
                                col_labels = list(view_sentiment.index), 
                                prefix_nm  = 'snt_')

    rich_sentiment_ = getTable(func = calcRichness, 
                                    data_lst   = [data[sh_cols]], 
                                    col_names  = list(view_sentiment.values), 
                                    col_labels = list(view_sentiment.index), 
                                    prefix_nm  = 'rh_snt_')
    agg_ = pd.concat([summi_sentiment_[0],rich_sentiment_[0]], axis = 1)
    return agg_

def get_distance_scores(data):
    data = get_meanvsh_scores(data)['score']
    sh_col_df = breakoutCols(data, txt = 'Sh')
    sh_cols = list(sh_col_df.index)
    
    # PANO SCORES-----------------------------------------------------------------------------
    #obs_condition = sh_col_df['Obstruction'] == 'unobs'
    view_distances = sh_col_df.fillna('sky').groupby('Distance').apply(lambda x: list(x.index))
    summi_dist_ = getTable(func = calcSummI, 
                                    data_lst = [data[sh_cols]], 
                                    col_names = list(view_distances.values), 
                                    col_labels = list(view_distances.index), 
                                    prefix_nm = 'sum_')
    summi_dist_2 = summi_dist_[0].copy()

    summi_dist_2 = (summi_dist_2.assign(sum_ShUne4 = (summi_dist_2['sum_ShUne4'] - data['ShUne4']),
                                        sum_ShSky  = data['ShUne4']))

    richness_dist = getTable(func = calcRichness, 
                                data_lst = [data[sh_cols]], 
                                col_names = list(view_distances.values), 
                                col_labels = list(view_distances.index),
                                prefix_nm = 'richness_')
    
    gini_dist = (round(summi_dist_[0],2)).apply(lambda x: qe.gini_coefficient(x.values), axis=1)
    gini_dist.name = 'dist_gini'
    # gini_dist.hist()

    pano_sum = summi_dist_2[['sum_ShFer3','sum_ShMit2','sum_ShUne4']].sum(1)
    pano_sum.name = 'pano_sum'
    pano_rich = richness_dist[0][['richness_ShFer3','richness_ShMit2','richness_ShUne4']].sum(1)- (data['ShUne4']>0)*-1
    pano_rich.name = 'pano_rich'

    unit_pano = pd.DataFrame(pano_sum.values / (1+pano_rich.values), index = pano_sum.index)
    unit_pano.columns = ['unit_pano']
    # unit_pano[unit_pano==0] = np.nan
    unit_pano = unit_pano.mask(unit_pano == 0)

    refuge_sum = summi_dist_[0][['sum_ShFer3','sum_ShMit2','sum_ShUne4']].sum(1) / summi_dist_[0].sum_ShNah1
    refuge_sum.name = 'refuge'

    # np.log1p(refuge_sum).hist(bins = 100, log = False)

    # unit_refuge = pd.DataFrame(refuge_sum.values / refuge_rich.values)
    # unit_refuge.columns = ['unit_refuge']
    # unit_refuge[unit_refuge==0] = np.nan   

    distance_scores = pd.concat([summi_dist_2,gini_dist,
                        pano_sum, pano_rich,
                        refuge_sum, 
                        # mystery_score1, mystery_score2, mystery_score3,
                        # unit_refuge, 
                        unit_pano
                        ],axis = 1)

    return distance_scores

def agg_mean_element(data):
    

    sh_col_df = breakoutCols(data, txt = 'Sh')
    sh_cols = list(sh_col_df.index)
    view_elements = sh_col_df.fillna('sky').groupby('Obstruction').apply(lambda x: list(x.index))
    summi_unobs_ = getTable(func = calcSummI, 
                                    data_lst = [data[sh_cols]], 
                                    col_names = list(view_elements.values), 
                                    col_labels = list(view_elements.index), 
                                    prefix_nm = 'meanvsh_')
    return summi_unobs_

def get_complexity_score(data):
    
    # summi_unobs_ = agg_mean_element(data) # 
    # data = get_meanvsh_scores(data)['score']
    data = agg_mean_element(data)[0].groupby(data.ID_Geb).mean()
    #COMPLEXITY SCORES
    richness_poi_ = calcRichness(data)
    richness_poi_.name = 'cmpx_rh'

    shanon_poi_ = calcShanonWeaver(data)
    shanon_poi_.name = 'cmpx_shanon'

    gini_poi = (round(data,2)).apply(lambda x: qe.gini_coefficient(x.values), axis=1)
    gini_poi.name = 'cmpx_gini'
    # gini_poi.hist()
    complexity_score = pd.concat([richness_poi_,shanon_poi_,gini_poi], axis = 1)
    return complexity_score