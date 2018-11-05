import pandas as pd
import requests
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.preprocessing import MinMaxScaler

np.random.seed(seed=13)
np.random.seed(seed=7)

def load_titanic_data(form="np", normalized=True):
    df = pd.read_csv('titanic.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    original_df = copy.deepcopy(df)
    original_df = original_df.sample(frac=1, random_state=13)
    original_df = original_df.drop(['Survived'], axis=1)

    # df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    df = pd.get_dummies(df, columns=['Embarked', 'Sex'], drop_first=True)
    # df = pd.get_dummies(df, columns=['Embarked','Sex'], drop_first=True)
    df = df.sample(frac=1, random_state=13)
    cutoff_train = 3*df.shape[0]//5
    cutoff_val = 4*df.shape[0]//5

    titanic_df = df.drop(['Survived'], axis=1)
    target_df = df['Survived']
    features = titanic_df.values
    labels = target_df.values

    X_train = features[:cutoff_val,:]
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_train = imp.fit_transform(X_train)
    y_train = labels[:cutoff_val]
    X_test = features[cutoff_val:,:]
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_test = imp.fit_transform(X_test)
    y_test = labels[cutoff_val:]
    # X_train1 = X_train[:cutoff_train,:]
    # y_train1 = y_train[:cutoff_train]
    # X_val = X_train[cutoff_train:,:]
    # y_val = y_train[cutoff_train:]

    scaler = MinMaxScaler().fit(X_train)
    Xscaled_train = scaler.transform(X_train)
    scaler_test = MinMaxScaler().fit(X_test)
    Xscaled_test = scaler_test.transform(X_test)

    if form == "np":
        if normalized:
            return Xscaled_train, y_train, Xscaled_test, y_test
        else:
            return X_train, y_train, X_test, y_test
    elif form == "df":
        return titanic_df, target_df
    else:
        return original_df

def load_tennis_data(form="np", normalized=True):
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_"
    url_endings = []
    for i in range(2014, 2017):
        url_endings.append(str(i)+".csv")
    dfs = []
    for e in url_endings:
        dfs.append(pd.read_csv(url+e))

    full_df = pd.concat(dfs)

    # Removing features
    full_df = full_df.drop(['tourney_id', 'draw_size', 'tourney_date', 'match_num', 'winner_id', 'loser_id', 'score', 'minutes'], axis=1)
    full_df = full_df.drop(['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'], axis=1)
    full_df = full_df.drop(['tourney_name', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc'], axis=1)
    full_df = full_df[np.isfinite(full_df['winner_rank'])]
    full_df = full_df[np.isfinite(full_df['loser_rank'])]
    full_df = full_df[np.isfinite(full_df['winner_rank_points'])]
    full_df = full_df[np.isfinite(full_df['loser_rank_points'])]
    full_df = full_df[np.isfinite(full_df['winner_ht'])]
    full_df = full_df[np.isfinite(full_df['loser_ht'])]
    full_df = full_df[np.isfinite(full_df['winner_age'])]
    full_df = full_df[np.isfinite(full_df['loser_age'])]
    # full_df = full_df.fillna('other')
    full_df['win'] = 1

    # Flipping examples to create negative examples
    flip_df = full_df.copy()
    flip_df['win'] = 0
    flip_df['winner_seed'] = full_df['loser_seed']
    flip_df['loser_seed'] = full_df['winner_seed']
    flip_df['winner_entry'] = full_df['loser_entry']
    flip_df['loser_entry'] = full_df['winner_entry']
    flip_df['winner_hand'] = full_df['loser_hand']
    flip_df['loser_hand'] = full_df['winner_hand']
    flip_df['winner_ht'] = full_df['loser_ht']
    flip_df['loser_ht'] = full_df['winner_ht']
    flip_df['winner_age'] = full_df['loser_age']
    flip_df['loser_age'] = full_df['winner_age']
    flip_df['winner_rank'] = full_df['loser_rank']
    flip_df['loser_rank'] = full_df['winner_rank']
    flip_df['winner_rank_points'] = full_df['loser_rank_points']
    flip_df['loser_rank_points'] = full_df['winner_rank_points']

    df = pd.concat([full_df, flip_df])
    column_dict = {}
    for col in list(df.columns):
        if 'winner' in col:
            column_dict[col] = col.replace('winner','p1')
        elif 'loser' in col:
            column_dict[col] = col.replace('loser','p2')
    df.rename(columns=column_dict, inplace=True)

    original_df = copy.deepcopy(df)
    original_df = original_df.sample(frac=1, random_state=7)
    original_df = original_df.drop(['win'], axis=1)
    dummy_columns = ['surface', 'p1_hand', 'p2_hand', 'p1_entry', 'p2_entry',
                    'round', 'tourney_level']
    # for d in dummy_columns:
    #     df[d] = df[d].fillna('other')
    # = pd.get_dummies(df, columns=dummy_columns, drop_first=True)
    cutoff_train = 3*df.shape[0]//5
    cutoff_val = df.shape[0]-1000
    df = df.sample(frac=1, random_state=7)
    tennis_df = df.drop(['win'] + dummy_columns + ['best_of'], axis=1)
    target_df = df['win']
    cols = tennis_df.columns

    # reduced_cols = ['p1_ht', 'p1_age', 'p1_rank', 'p1_rank_points', 'p2_ht', 'p2_age', 'p2_rank', 'p2_rank_points']
    # series_list = []
    # for col in reduced_cols:
    #     series_list.append(tennis_df[col])
    # reduced_tennis_df = pd.concat(series_list, axis=1)

    # Create train, validation, test sets
    # features = reduced_tennis_df.values
    features = tennis_df.values
    labels = target_df.values

    X_train = features[:cutoff_val,:]
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_train = imp.fit_transform(X_train)
    y_train = labels[:cutoff_val]
    X_test = features[cutoff_val:,:]
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_test = imp.fit_transform(X_test)
    y_test = labels[cutoff_val:]
    # X_train1 = X_train[:cutoff_train,:]
    # y_train1 = y_train[:cutoff_train]
    # X_val = X_train[cutoff_train:,:]
    # y_val = y_train[cutoff_train:]

    scaler = MinMaxScaler().fit(X_train)
    Xscaled_train = scaler.transform(X_train)
    scaler_test = MinMaxScaler().fit(X_train)
    Xscaled_test = scaler_test.transform(X_test)

    if form == "np":
        if normalized:
            return Xscaled_train, y_train, Xscaled_test, y_test
        else:
            return X_train, y_train, X_test, y_test
    elif form == "df":
        return tennis_df, target_df
    else:
        return original_df
