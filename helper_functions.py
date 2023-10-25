import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import r2_score
from scipy.stats import binom, poisson, nbinom, expon, gamma
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import warnings
import statsmodels.api as sm
from sklearn import linear_model


#generic train test split by competition
def train_val_split_by_competition(data, train_proportion=0.7, return_separate=False):
    data.sort_values(["datetime"], inplace = True)

    def split_dataframe(group):
        group.sort_values(["datetime"], inplace = True)
        n = len(group)
        split_index = int(train_proportion * n)
        return group[:split_index], group[split_index:]

    grouped_data = data.groupby(["league_name"]).apply(split_dataframe)

    if return_separate==False:
        train_data = pd.concat([group[0] for group in grouped_data], axis = 0)
        val_data = pd.concat([group[1] for group in grouped_data], axis = 0)
    else:
        train_data = [group[0] for group in grouped_data]
        val_data = [group[1] for group in grouped_data]
    
    return train_data, val_data

#get_week_difference
def get_week_difference(data, current_year, current_week):
    def func(row):
        d = (current_year - row["year"]) * 52 + (current_week - row["week"])
        assert(d >= 0)
        return d
        
    diff = data.apply(func, axis = 1)
    return diff

#get_weights
def get_weights(week_diff, decay_factor):
    return np.exp(-decay_factor * week_diff)

#get weekly and all previous matches 
def split_dataset(data, year, week):
    train_train = data[(data.year < year) | ((data.year == year) & (data.week < week))]
    train_test = data[(data.year == year) & (data.week == week)]

    return train_train, train_test

#encode features
def one_hot_encode_data(df, encoding_dict, preprocessor=None):
    if preprocessor == None:
        #Create a list to hold all transformers
        transformers = []
        for col, (drop_method, prefix) in encoding_dict.items():
            # Create a transformer for each column in the encoding_dict
            transformer = OneHotEncoder(drop=drop_method, handle_unknown='ignore',sparse_output=False)
            transformers.append((f'onehot_{col}', transformer, [col]))

        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # This will keep the other columns unchanged
        )
        # Fit and transform the data
        df_transformed = preprocessor.fit_transform(df)
        
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df_transformed = preprocessor.transform(df)
    

    # Get feature names from the transformers
    feature_names = []
    for col, (_, prefix) in encoding_dict.items():
        feature_names += [f"{prefix}_{feature.split('_')[-1]}" for feature in preprocessor.named_transformers_[f'onehot_{col}'].get_feature_names_out([col])]

    # Add the other columns to the list of feature names
    other_cols = [col for col in df.columns if col not in encoding_dict]
    all_feature_names = feature_names + other_cols

    # Create a new DataFrame from the transformed data
    df_transformed = pd.DataFrame(df_transformed, columns=all_feature_names)
    df_transformed = df_transformed[other_cols + feature_names]
    
    return df_transformed, preprocessor

#standardize
def standardize(X, cat_vars, scaler=None):
    X_cont = X.drop(cat_vars, axis = 1)
    
    if scaler==None:
        scaler = StandardScaler()
        scaler.fit(X_cont)
    
    X_cont = pd.DataFrame(scaler.transform(X_cont), columns = X_cont.columns)
    x_cat = X[cat_vars].copy().reset_index(drop = True)
    X = pd.concat([X_cont, x_cat], axis = 1)
    return X, scaler

#get poisson log likelihoods given expectancies
def get_log_likelihood(y_true, parameters, distribution='Poisson',individual_scores=False):
    assert(len(y_true) == len(parameters[0]))
    if distribution =='Poisson' or distribution == 'Exponential':
        assert(len(parameters) == 1)
    elif distribution =='Gamma':
        assert(len(parameters) == 2)
    
    if distribution=='Poisson':
        expectancies = parameters[0]
        probs = poisson.pmf(y_true, expectancies)
        ind_log_ll = np.log(probs)
        
    elif distribution=='Exponential':
        expectancies = parameters[0]
        probs = expon.pdf(y_true, scale= 1/expectancies)
        ind_log_ll = np.maximum(-10000,np.log(probs))
        
    elif distribution=='Gamma':
        scales = parameters[0]
        shapes = parameters[1]
        probs = gamma.pdf(y_true, a = shapes,scale=scales)
        ind_log_ll = np.maximum(-10000,np.log(probs))
        
    else:
        raise ValueError('Invalid distribution argument passed.')
        
    
    log_ll = np.sum(ind_log_ll)
    avg_log_ll = log_ll/len(probs)
    
    if individual_scores == False:
        return log_ll, avg_log_ll
    else:
        return ind_log_ll
    
#estimate scale parameter from y_true, predicted means (in-sample) and dof residuals 
# shape_from_model = 1/results.scale
# mean_scale = preds.mean()/shape_from_model
# scales = [m_i /shape_from_model for m_i in preds]
# shape_from_model, mean_scale
#see 'https://www.statsmodels.org/stable/_modules/statsmodels/genmod/generalized_linear_model.html#GLM.estimate_scale'
def estimate_x2_scale(y, mu, n_sample, dof):
    resid = np.power(y- mu, 2)
    variance = mu ** 2
    df_residuals = n_sample - dof
    return np.sum(resid / variance) / df_residuals

#get consecutive week train-evaluation
def get_consecutive_evaluations(pre_train_dat,dat, decay_factor, reg_param, features_to_use,target_var,
                                cat_vars, encoding_dict, distribution, fit_intercept=False, lower_thresh=0,verbose=False):

    df = dat.copy()
    df = df.sort_values(["year", "week"])
    grouped = df.groupby(["year", "week"], as_index = False)
    if distribution == "Gamma":
        cols = ["date","squad", "player","expectancies", "scale","shape", "target"]
    elif distribution == "Poisson":
        cols = ["date","squad", "player","expectancies","target"]
    else:
        raise Exception("Invalid distribution argument.")
        
    exps = pd.DataFrame(columns=cols)
    counter = 0
    total_samples = 0
    
    for i,_ in grouped:
        if counter >= lower_thresh:
            year = i[0]
            week = i[1]
            if verbose == True:
                print(f"Year = {year}, week = {week}...")

            #split validation data by week
            val_train, val_test = split_dataset(df, year, week)
            if pre_train_dat is None:
                data_to_train = val_train
            else:
                data_to_train = pd.concat([pre_train_dat, val_train], axis = 0)
            
            if decay_factor != None:
                week_diff = get_week_difference(data_to_train, year, week)
                weights = get_weights(week_diff, decay_factor)
            else:
                weights = None

            #get x_train,x_test,y_train,y_test
            x_train = data_to_train[features_to_use].reset_index(drop=True)
            y_train = data_to_train[target_var].values

            x_test = val_test[features_to_use].reset_index(drop=True)
            y_test = val_test[target_var].values

            #standardize
            x_train, scaler = standardize(x_train, cat_vars=cat_vars)
            x_test, _ = standardize(x_test, cat_vars=cat_vars, scaler=scaler)

            #one hot encode x_train and x_test
            x_train, encoder = one_hot_encode_data(x_train, encoding_dict=encoding_dict)
            x_test, _ = one_hot_encode_data(x_test, encoding_dict=encoding_dict, preprocessor=encoder)
            
            #add constants if fit_intercept == True
            if fit_intercept == True:
                x_train = sm.add_constant(x_train)
                x_test = sm.add_constant(x_test)

            #get features
            features = x_train.columns
            
            #train model
            if distribution=="Gamma":
                model = linear_model.GammaRegressor(fit_intercept=False, alpha = reg_param, max_iter = 500)
                model.fit(x_train, y_train, sample_weight = weights)
                
                train_preds = model.predict(x_train)
                preds = model.predict(x_test)
                #get in-sample shape parameter
                inv_shape_param =  estimate_x2_scale(y_train, train_preds, len(x_train), len(features))
                shape_from_model = 1/inv_shape_param

                scales = [m_i /shape_from_model for m_i in preds]

                ids = val_test[["date","squad", "player"]].reset_index(drop=True)
                ids["expectancies"] = preds
                ids["scale"] = scales
                ids["shape"] = shape_from_model
                
            elif distribution == "Poisson":
                model = linear_model.PoissonRegressor(fit_intercept=False, alpha = reg_param, max_iter = 500)
                model.fit(x_train, y_train, sample_weight = weights)
                
                preds = model.predict(x_test)
                ids = val_test[["date","squad", "player"]].reset_index(drop=True)
                ids["expectancies"] = preds
                
            else:
                raise Exception("Invalid distribution arg.")
                                
            coefs = model.coef_
            coef_dict = dict(zip(features, coefs))
            ids["target"] = y_test
            
            exps = pd.concat([exps, ids], axis = 0)
            n_samples_train = len(x_train)
            n_samples_test = len(x_test)
            total_samples += n_samples_test 
            
        counter += 1

    print(f"total val samples = {total_samples}")
    return exps

#tune hyperparameters
def tune_hyperparameters(dat,reg_grid, decay_grid, features_to_use,target_var,
                         cat_vars, encoding_dict, distribution, lower_thresh=5):
    
    grid_scores = []
    for reg in reg_grid:
        for dec in decay_grid:
            print(f"Trying reg={reg} and decay factor = {dec}")
            model_results = get_consecutive_evaluations(None,dat, dec, reg,features_to_use,target_var,
                                                        cat_vars,encoding_dict,distribution,
                                                        lower_thresh=lower_thresh)
            
            
            if distribution == "Gamma":
                params = (model_results.scale, model_results["shape"])
            else:
                params = (model_results.expectancies,)
            
            llres, avg_llres = get_log_likelihood(model_results.target,
                                       params, 
                                       distribution=distribution,
                                       individual_scores=False)
            
            r2score  = r2_score(model_results.target, model_results.expectancies)
            grid_scores.append((reg, dec, r2score, llres, avg_llres))
            
    grid_scores = pd.DataFrame(grid_scores, columns=["reg_parameter","decay_parameter","r2score","logl_exp","avg_logl_exp"])
    return grid_scores