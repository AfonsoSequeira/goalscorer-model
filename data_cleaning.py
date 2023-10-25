import pandas as pd
import numpy as np
import os

league_code_dict = {
    '9':'Premier League',
    '10':'Championship',
    '11':'Serie A',
    '12':'La Liga',
    '13':'Ligue 1',
    '20':'Bundesliga',
    '32':'Primeira Liga',
    '23':'Eredivisie'
}

#convert datetime
def add_datetime(data):
    data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
    return data

def load_data(seasons_to_load, leagues_to_load):
    base_dir = "C:/MyDevelopment/Goalscorers/fbref_data/event_data/"
    df_summary = []
    df_possession = []
    
    for file in os.listdir(base_dir):
        if "summary" in file or "possession" in file:
            fh = file.split("-")
            season = "-".join(fh[0:2])
            league_code = fh[3]
            league = league_code_dict[league_code]
            
            if (leagues_to_load is None or league_code in leagues_to_load) and (seasons_to_load is None or season in seasons_to_load):
                season_data = pd.read_csv(base_dir+file)
                season_data["league_name"] = league
                season_data["season"] = season
                if "summary" in file:
                    df_summary.append(season_data)
                else:
                    df_possession.append(season_data)
        
    event_data = add_datetime(pd.concat(df_summary, axis=0))
    possession_data = add_datetime(pd.concat(df_possession, axis=0))

    common_cols = [col for col in event_data.columns if col in possession_data.columns]
    data = pd.merge(event_data, possession_data, how='inner', on=common_cols, validate='1:1')

    return data

#get opposing team
def add_opposite_team(data):
    data["squad_opp"] = np.where(data["squad"]==data["home_team"], data["away_team"], data["home_team"])
    return data

#map position
def map_position(data):
    pos_map = {
        'CM':'CM',
        'LB':'FB',
        'RB':'FB',
        'DF':'CB',
        'MF':'CM',
        'RW':'W',
        'LW':'W',
        'LM':'CM',
        'RM':'CM',
        'FW':'FW',
        'CB':'CB',
        'DM,CM':'DM',
        'CM,DM':'CM',
        'DM,CM':'DM',
        'RM,CM':'CM',
        'LW,LM':'W',
    }

    def map_to_generic_position(pos,pos_map):
        if "," in pos:
            f_pos = pos.split(",")[0]
        else:
            f_pos = pos

        try:
            return pos_map[f_pos]
        except:
            return f_pos
    
    data["raw_position"] = data["position"]
    data["position"] = data["position"].astype(str).transform(lambda x : map_to_generic_position(x, pos_map))
    data = data[data["position"] != 'nan']
    
    return data

#drop NAs in npxg, xg and minutes
def drop_NAs(data):
    data = data[~((data.npxg.isnull()) | (data.xg.isnull()) | (data.minutes.isnull()))]
    return data

#remove gk
def remove_gk(data):
    data = data[data["position"] != 'GK']
    return data

#add supremacy
def add_supremacy(data):
    data["supremacy"] = data["goal_exp"] - data["goal_exp_opp"]
    return data

#get non-penalty goals
def add_npg(data):
    data["npg"] = data.goals - data.pens_made
    return data

#add year and week 
def add_year_week(data):
    data["year"] = data["datetime"].dt.isocalendar().year
    data["week"] = data["datetime"].dt.isocalendar().week
    
    return data

#add expectancies
def add_goal_expectancies(data, goal_exp):
    data['date'] = pd.to_datetime(data.datetime.dt.date)
    goal_exp.date = pd.to_datetime(goal_exp.date)
    data = data.merge(goal_exp, on=["home_team", "away_team", "date"], how='inner')

    #get exp and exp_opp
    data["goal_exp"] = np.where(data.squad == data.home_team, data.home_exp, data.away_exp)
    data["goal_exp_opp"] = np.where(data.squad == data.home_team, data.away_exp, data.home_exp)

    return data

def add_npxg_per_minute(data):
    data["npxg_per_min"] = data["npxg"]/data["minutes"]
    return data

def add_team_scored_and_conceded_npxg(data):
    team_npxg = data.groupby(["datetime","squad"], as_index=False)["npxg"].sum().rename(columns={'npxg':'team_npxg'})
    data = pd.merge(data, team_npxg, on=["datetime","squad"], validate='m:1')

    team_npxg.rename(columns={'squad':'squad_opp','team_npxg':'team_conceded_npxg'}, inplace=True)
    data = pd.merge(data, team_npxg, on=["datetime", "squad_opp"], validate='m:1')
    
    return data

def add_solo_striker_position(data):
    start_data = data[data.start == True].copy(deep=True)
    start_data['is_FW'] = (start_data['position'] == 'FW').astype(int)
    n_strikers_data = start_data.groupby(['datetime', 'squad'], as_index=False)['is_FW'].sum().rename(columns={'is_FW':'n_FW'})

    data = pd.merge(data, n_strikers_data, on=['datetime', 'squad'], validate='m:1')
    data["position"] = np.where((data["position"] == 'FW') & (data["n_FW"] == 1), "S_FW", data["position"])
    
    return data

#if two goalkeepers have played for the same team, we pick the one with the most minutes in that match
def add_main_opposing_gk(data):
    gk_data = data[data.position == 'GK']
    max_minutes_idx = gk_data.groupby(['date', 'squad'])['minutes'].idxmax()
    gk_data = gk_data.loc[max_minutes_idx][["date","home_team","away_team","squad","player"]].reset_index(drop=True).rename(columns={'player':'main_gk'})

    data = pd.merge(data, gk_data, on=["date","home_team","away_team"], suffixes=('', '_gk'))
    data = data[data.squad != data.squad_gk]
    data = data.drop(["squad_gk"],axis=1)
    data = data.rename(columns={'main_gk':'gk_opp'})
    
    return data

if __name__ == '__main__': 
    data = load_data(None,None)