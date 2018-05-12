import warnings
warnings.filterwarnings('ignore')

import tqdm
import h5py
import pickle as pk
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import scipy.stats
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from backtester import calc_score, apply_on_test, test_generator, make_submission
import utils

# f1 and f2 -- Files with training set, f3 -- with test set
f1 = h5py.File('part1.hdf5', 'r')
f2 = h5py.File('part2.hdf5', 'r')
f3 = h5py.File('part3.hdf5', 'r')

# Pair for prediction
PAIR_NAMES = ['BTC_USDT']
# To reduce dataset, we left only relevant currency
RELEVANT_CURRENCY = 'BTC'
TARGET_COLUMNS = sorted(["hitbtc/" + pair + "/buy/max_price" for pair in PAIR_NAMES])

TICKS_IN_FIVE_MINUTES = 5 * 60 // 10
# Horizon of the forecast. This parameter is not recommended to change
PREDICT_DELAY_TICKS = 30
# GAP determines the sampling frequency of objects in the training set. 
# To reduce the required memory, this parameter can be increased
GAP = 2

# At this cell primary feature selection (relevant_columns)
RELEVANT_COLUMNS = []
header = [name for name in f1["header"]]
for exch, symbol, side, candles in tqdm.tqdm(utils.candles_loop(f1["body"])):
    for name in header:
        # "hitbtc/BTC_USDT/buy/max_price"
        column = exch + "/" + symbol + "/" + side + "/" + name
        if RELEVANT_CURRENCY in symbol and name in ["open_price", "min_price", "max_price", "amount", "number_of_trades"] \
                                       and exch in ['hitbtc', 'bitfinex2']:
            RELEVANT_COLUMNS += [column]
header = f1["header"][:]

def custom_spearman(y, preds):
    score, _ = scipy.stats.spearmanr(y, preds)
    return score
    #return 'spearman', score, True

def custom_spearman2(y, preds):
    score, _ = scipy.stats.spearmanr(y, preds)
    return 'spearman', score, True

def load_X_y(f, target, relevant):
    exch, pair, action, _ = TARGET_COLUMNS[0].split("/")
    loaded = np.zeros((len(f["body"][exch][pair][action]), 0))
    columns = []
    for exch, symbol, side, candles in tqdm.tqdm(utils.candles_loop(f["body"])):
        # To reduce dataset, select only relevant columns
        for feature_num, name in enumerate(header):
            column = exch + "/" + symbol + "/" + side + "/" + str(name)
            if column in RELEVANT_COLUMNS:
                loaded = np.column_stack((loaded, candles[:, feature_num]))
                columns += [exch + "/" + symbol + "/" + side + "/" + name]

    set_tq = set(target)
    pairs_idx = []
    for i, column in enumerate(columns):
        if column in set_tq:
            pairs_idx += [i]
                            
    # for backtester comatability
    loaded = pd.DataFrame(loaded, columns=columns)[RELEVANT_COLUMNS].values
    
    return loaded, pairs_idx

def get_X_y(loaded, targe_idx):
    X = []
    y = []
    
    for start in tqdm.tqdm(range(0, len(loaded) - TICKS_IN_FIVE_MINUTES - PREDICT_DELAY_TICKS, GAP)):
        end = start + TICKS_IN_FIVE_MINUTES
        X += [(loaded[start+1:end] - loaded[start:end-1]).flatten()]
        y += [loaded[end-1 + PREDICT_DELAY_TICKS, targe_idx] / loaded[end-1, targe_idx]  - 1]
    X = np.array(X, dtype='float32')   
    y = np.array(y, dtype='float32')
    return X, y


class Model:
    def __init__(self, pairs, relevant_columns):
        """
        :param pairs:
        """
        self.pairs = pairs
        # List of RELEVANT_CURRENCY
        assert len(self.pairs) == 1
        assert "BTC_USDT" in self.pairs
        
        self.relevant_columns = relevant_columns
        
        with open("BTC_USDT_2.pk", "rb+") as f:
            self.regr = pk.load(f)
    
    def __extract_features(self, test):
        relevant_test_features = test[self.relevant_columns]
        return (relevant_test_features[1:].values - relevant_test_features[:-1].values).flatten()
    
    def __predict_pair(self, test, pair):
        """

        :param test: dataframe to predict on
        :return: predicted ratio
        """
        if pair not in self.pairs:
            return 0.
        
        features = self.__extract_features(test)
        
        return self.regr.predict([features])[0]
    
    def predict(self, test):
        """
        :param test: dataframe to predict on
        :return: dict: pair -> predicted ratio
        """
        output = {}
    
        for pair in self.pairs:
            output[pair] = self.__predict_pair(test, pair)
        
        return output


loaded1, target_idx = load_X_y(f1, TARGET_COLUMNS, RELEVANT_CURRENCY)
loaded2, _ = load_X_y(f2, TARGET_COLUMNS, RELEVANT_CURRENCY)
loaded3, _ = load_X_y(f3, TARGET_COLUMNS, RELEVANT_CURRENCY)

f1.close()
f2.close()
f3.close()

X_train1, y_train1 = get_X_y(loaded1, target_idx)
X_train2, y_train2 = get_X_y(loaded2, target_idx)
X_val, y_val = get_X_y(loaded3, target_idx)
X_train = np.vstack((X_train1, X_train2))
y_train = np.concatenate((y_train1, y_train2))
del(X_train1, X_train2, y_train1, y_train2)

def find_best_lgbm_regr(X_train, y_train):
    # Fit simple regression model
    my_score_func = make_scorer(score_func=custom_spearman, greater_is_better=True)
    param_grid = [
        {'learning_rate' : [0.01], 'num_leaves': [15], 'reg_alpha': [0.5, 1.0]},
    #    {'learning_rate' : [0.01], 'num_leaves': [15], 'reg_lambda': [1.0]}
    ]

    lgbm_regr = lgb.LGBMRegressor(n_estimators=1000, silent=False, n_jobs=4, random_state=2018)
    grid_search = GridSearchCV(estimator=lgbm_regr, param_grid=param_grid, cv=2, scoring=my_score_func, verbose=2)
    grid_search.fit(X=X_train, y=y_train.ravel())

    print(grid_search.best_params_, grid_search.best_score_)
    best_lgbm_regr = grid_search.best_estimator_
    return best_lgbm_regr

def try_some_lgbm_regr(X_train, y_train, X_val, y_val):
    lgbm_regr = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01,
                                  num_leaves=15, reg_alpha=0.5,
                                  silent=False, n_jobs=4, random_state=2018)
    lgbm_regr.fit(X_train, y_train.ravel(), eval_set=(X_val, y_val.ravel()),
             eval_metric=custom_spearman2, early_stopping_rounds=500)
    return lgbm_regr


lgbm_regress = find_best_lgbm_regr(X_train, y_train)
#lgbm_regress = try_some_lgbm_regr(X_train, y_train, X_val, y_val)
y_val_pred = lgbm_regress.predict(X_val)
print("Spearman index on testset = ", spearmanr(y_val_pred, y_val.ravel()))

# dump the best model
f = open('BTC_USDT_2.pk', 'wb+')
pk.dump(lgbm_regress, f)
f.close()

model = Model(['BTC_USDT'], RELEVANT_COLUMNS)

y_true, y_pred, pairs = apply_on_test(model, 'part3.hdf5')
calc_score(y_true, y_pred, pairs)
make_submission(y_pred, pairs, "baseline_2_part3.csv")
