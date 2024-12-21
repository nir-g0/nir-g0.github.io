import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import teamestimatedmetrics, playergamelog, PlayerProfileV2
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import requests
from datetime import date
import joblib
import tensorflow as tf
import sys
from k_fold_model import train_k_fold
from k_fold_nn import train_k_fold_nn


def get_player_info(player_name):
    """Retrieve player information by name."""
    try:
        player = players.find_players_by_full_name(player_name)[0]
        return player
    except IndexError:
        print("Player not found, please check your spelling.")
        sys.exit(1)


def get_recent_game_averages(player_id, player_features):
    """Retrieve recent game averages for the player."""
    df = pd.DataFrame(playergamelog.PlayerGameLog(player_id).get_data_frames()[0])
    df = df.iloc[0:6, :]
    Q1 = df['PTS'].quantile(0.25)
    Q3 = df['PTS'].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['PTS'] >= lower_bound) & (df['PTS'] <= upper_bound)]
    df.to_csv('no_outliers.csv')
    df = df[player_features]
    
    avg_df = pd.DataFrame()
    for feature in player_features:
        avg_df[feature] = [np.average(df[feature])]
    return avg_df


def get_todays_games():
    """Retrieve today's games from the NBA schedule."""
    r = requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json")
    data = r.json()

    today = str(date.today())
    today = today.split('-')
    today = f'{today[1]}/{today[2]}/{today[0]}'

    for date_entry in data['leagueSchedule']['gameDates']:
        if today in date_entry['gameDate']:
            return date_entry['games']
    return []


def find_opponent_team(player_team_abr, todays_games):
    """Find the opponent team for the given player team abbreviation."""
    for game in todays_games:
        home = game['homeTeam']
        away = game['awayTeam']
        if home['teamTricode'] == player_team_abr:
            return away['teamTricode'], home['teamId']
        elif away['teamTricode'] == player_team_abr:
            return home['teamTricode'], away['teamId']

    print("Player not playing today, check line tomorrow.")
    sys.exit(0)


def get_opponent_metrics(op_team_id):
    """Retrieve opponent team's defensive rating and pace."""
    all_teams_metrics = pd.DataFrame(teamestimatedmetrics.TeamEstimatedMetrics().get_data_frames()[0])
    for i in range(len(all_teams_metrics.index)):
        if op_team_id == all_teams_metrics.iloc[i, :]['TEAM_ID']:
            return all_teams_metrics.iloc[i, :]['E_DEF_RATING'], all_teams_metrics.iloc[i, :]['E_PACE']
    return None, None


def add_interaction_terms(avg_df):
    """Add interaction terms to the feature dataframe."""
    avg_df['FGM_FGA_Interaction'] = avg_df['FGM'] * avg_df['FGA']
    avg_df['FTM_FG_PCT_Interaction'] = avg_df['FTM'] * avg_df['FG_PCT']
    return avg_df


def scale_features(avg_df):
    """Scale features using the saved scaler."""
    scaler = joblib.load('scaler.pkl')
    return scaler.transform([avg_df.values[0]])


def load_model_and_predict_ridge(scaled_features):
    """Load Ridge regression model and make predictions."""
    ridge_model = np.load('parlay_prophet_model_k_fold.npz')
    coefficients = ridge_model['coefficients']
    intercept = ridge_model['intercept']

    def predict(X_new, coefficients, intercept):
        return np.dot(X_new, coefficients) + intercept

    return predict(scaled_features, coefficients, intercept)[0]


def load_nn_model_and_predict(scaled_features):
    """Load a neural network model (.keras) and make predictions."""
    nn_model = tf.keras.models.load_model('parlay_prophet_model_nn.keras')
    return nn_model(scaled_features)[0][0]


def calculate_confidence_interval(predicted_points, mae, confidence_level=0.95):
    """Calculate the confidence interval for predictions."""
    z_score = norm.ppf((1 + confidence_level) / 2)
    return predicted_points - z_score * mae, predicted_points + z_score * mae


def make_recommendation(predicted_points, betting_line, lower, upper):
    """Generate a betting recommendation based on the prediction."""
    if predicted_points > betting_line + 1 and lower > betting_line:
        return "OVER"
    elif predicted_points < betting_line - 1 and upper < betting_line:
        return "UNDER"
    else:
        return "No bet, projected too close to line"


if __name__ == "__main__":
    # Define inputs

    player_name = sys.argv[1]
    betting_line = float(sys.argv[2])

    # Define player features
    player_features = ['FGM', 'FGA', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT', 'FTM']

    # Retrieve player info
    player = get_player_info(player_name)

    # Get recent game averages
    avg_df = get_recent_game_averages(player['id'], player_features)

    # Get today's games and find opponent
    todays_games = get_todays_games()
    player_team_abr = str(PlayerProfileV2(player_id=player['id']).get_data_frames()[-1]['PLAYER_TEAM_ABBREVIATION'].values[0])
    upcoming_opponent, op_team_id = find_opponent_team(player_team_abr, todays_games)

    # Get opponent metrics
    op_def_rating, op_pace = get_opponent_metrics(op_team_id)
    avg_df['OP_DRTG'] = op_def_rating
    avg_df['OP_PACE'] = op_pace

    # Add interaction terms
    avg_df = add_interaction_terms(avg_df)

    # Scale features
    scaled_features = scale_features(avg_df)

    # Train model and get MAE
    mae = train_k_fold()
    mae_2 =  train_k_fold_nn()

    # Predict points with Ridge regression
    predicted_points_ridge = load_model_and_predict_ridge(scaled_features)

    # Predict points with Neural Network
    predicted_points_nn = load_nn_model_and_predict(scaled_features)

    # Confidence interval for Ridge
    confidence_interval_ridge = calculate_confidence_interval(predicted_points_ridge, mae)
    
    confidence_interval_nn = calculate_confidence_interval(predicted_points_ridge, mae_2)
    # Recommendation
    points_avg = (predicted_points_ridge+predicted_points_nn)/2
    lower_bound  = (confidence_interval_nn[0]+ confidence_interval_ridge[0])/2
    upper_bound  = (confidence_interval_nn[1]+confidence_interval_ridge[1])/2
    recommendation = make_recommendation(points_avg, betting_line, lower_bound, upper_bound)

    # Print results
    print(f"Model 1: {predicted_points_ridge:.2f} points against {upcoming_opponent} (DEF rating {op_def_rating})")
    print(f"Likely Range (Model 1): [{confidence_interval_ridge[0]:.2f} to {confidence_interval_ridge[1]:.2f}]")
    
    print(f"Model 2: {predicted_points_nn:.2f}")
    print(f"Likely Range (Model 2): [{confidence_interval_nn[0]:.2f} to {confidence_interval_nn[1]:.2f}]")
    
    print(f"Recommendation: {recommendation}")