import pandas as pd

#get rolling avg/sum variables
def add_player_avg_feature(data, var_col, window_size):
    f = lambda x: x.rolling(window_size, min_periods=1).mean()  

    data = data.sort_values("datetime")
    data['shited_col'] = data.groupby("player_id", as_index=False)[var_col].transform(lambda x: x.shift(1))
    data[f'avg_{var_col}_l{window_size}'] = data.groupby("player_id", as_index=False)['shited_col'].transform(f).fillna(0)
    data.drop(['shited_col'], axis = 1, inplace = True)
    
    return data

def add_team_avg_feature(data, var_col, window_size):
    f = lambda x: x.rolling(window_size, min_periods=1).mean()  
    
    team_data = data[["datetime","squad", var_col]].drop_duplicates()#get unique team rows
    team_data = team_data.sort_values("datetime")
    team_data['shited_col'] = team_data.groupby("squad", as_index=False)[var_col].transform(lambda x: x.shift(1))
    team_data[f'avg_{var_col}_l{window_size}'] = team_data.groupby("squad", as_index=False)['shited_col'].transform(f).fillna(0)
    team_data.drop(['shited_col'], axis = 1, inplace = True)
    
    #merge back with all data
    data = pd.merge(data, team_data, on=["datetime","squad", var_col], validate='m:1')
    
    return data, team_data