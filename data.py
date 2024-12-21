import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelog, teamestimatedmetrics

player_list = []

def get_player_stats(input):
        global player_list 
        if len(player_list) == 0 :
            get_all_players()
        if input.capitalize() not in player_list:
            print('Invalid Player')
            return
        player = players.find_players_by_full_name(input)[0]
        player_id = player['id']

        career = playergamelog.PlayerGameLog(player_id=player_id)
        return career.get_data_frames()[0]
        
def get_all_players():
    global player_list 
    for obj in players.get_active_players():
        player_list.append(obj['full_name'].capitalize())
        
def find_team_row(dataframe, id):
    for i in range(0,30):
        if(dataframe.iloc[i,1] == id):
            return i
    return -1

def extend_data(dataframe, metrics):
    home_game = []
    defense_rating = []
    pace_rating = []
    size = len(dataframe.index)
    for i in range(0,size):
        matchup = dataframe.iloc[i,-1].replace(' ', '', -1)
        home = 1
        if('@' in matchup):
            home = 0
            matchup = matchup.split('@')[1]
        else:
            matchup = matchup.split('vs.')[1]
        team = teams.find_team_by_abbreviation(matchup)
        row = metrics.iloc[find_team_row(metrics, team['id'])]
        
        home_game.append(home)
        defense_rating.append(row['E_DEF_RATING'])
        pace_rating.append(row['E_PACE'])
        
    dataframe = dataframe.assign(HOME_GAME=home_game,OP_DRTG=defense_rating,OP_PACE=pace_rating)
    dataframe.drop('MATCHUP', axis=1, inplace=True)
    return dataframe

def process_query(query):
    
    features = ['FGM',  'FGA',  'FG_PCT',  'FG3M',  'FG3A',  'FG3_PCT',  'FTM',  'FTA', 'FT_PCT']
    supervisor = 'PTS'
    metrics = teamestimatedmetrics.TeamEstimatedMetrics()
    metrics = pd.DataFrame(metrics.get_data_frames()[0])
    # Find player by name
    player = get_player_stats(query)
    df = pd.DataFrame(player)
    output = df[supervisor]

    features_plus = features.copy()
    features_plus.append('MATCHUP')

    df = df[features_plus]
    df = extend_data(df, metrics)
    df['FGM_FGA_Interaction'] = df['FGM'] * df['FGA']
    df['FTM_FG_PCT_Interaction'] = df['FTM'] * df['FG_PCT']
    df['Points'] = output

    # df = df.iloc[0:11,:]

    df.to_csv('processed_stats.csv')