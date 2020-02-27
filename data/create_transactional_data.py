"""Example code to generate and convert transactional data to i.i.d. data."""

"""
1. Run data recipe under "+ ADD DATASET" import dialog to create transactional data
2. Run as data recipe under "raw_transactions_non_iid" -> "DETAILS" -> "MODIFY BY RECIPE"
3. Run experiments on i.i.d. train/test splits (Note: leaky features are for instructional purposes only)
"""

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from h2oaicore.data import CustomData  # BoilerPlate - not needed for unit test

import datatable as dt
import numpy as np
import pandas as pd

# global definitions: please customize
col_date = "time"  # time column name
col_group = "customer_id"  # grouping column name (customer or similar)
target = 'target'  # target column name
target_labels = ['No', 'Yes']  # class labels
col_row_id = "__internal_row_id__"  # helper to make sure row order is preserved

# transactions characteristics: please customize
n_customers = 100
min_transactions = 10
max_transactions = 100
min_date = '2019-01-01'
split_date = '2019-06-30'
max_date = '2019-08-31'

# feature creation: please customize
make_features_from_scratch = False
window_length_days = [3, 5, 7, 10, 14]
operators = ["mean", "max", "sum"]
shuffle = True
leaky_choices = [True, False]  # True creates leaky features and data splits - for instructinoal purposes only!


class TransactionalToIID(CustomData):
    """ Converts transactional data to i.i.d. data"""

    @staticmethod
    def make_transactions():
        """ Helper to creates transactional data if used as data creation recipe"""
        np.random.seed(15)

        # create a list of random timestamps
        def create_ts_list(start, end, n):
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            ndays = (end - start).days + 1
            return pd.to_timedelta(np.random.rand(n) * ndays, unit='D') + start

        df = []

        # Create dataset for all dates and make sure there is a relation between train and valid targets
        customer_list = np.random.randint(0, 10 * n_customers, n_customers)
        length = 0
        for i, c in enumerate(customer_list):
            ts_list = create_ts_list(min_date, max_date, np.random.randint(min_transactions, max_transactions))
            # print(ts_list)
            # print(ts_list.shape)
            new_row = pd.DataFrame(
                {
                    col_group: (np.ones(len(ts_list)) * c).astype(int),
                    col_date: ts_list,
                    # add some random features
                    'x1': np.random.randint(low=0, high=7, size=len(ts_list)),
                    'x2': np.random.normal(size=len(ts_list)),
                    target: np.random.choice([0, 1], p=[0.8, 0.2], size=len(ts_list))
                }
            )
            length += new_row.shape[0]
            # Make target depend on previous target values using a rolling average
            new_row[target] = (new_row[target].rolling(15, min_periods=1).mean() > .3).astype(int)
            new_row[target] = new_row[target].map({0: target_labels[0], 1: target_labels[1]})

            df.append(new_row)

        X = pd.concat(df, axis=0)
        X.reset_index(drop=True, inplace=True)
        X = X.sort_values([col_group, col_date])
        X[col_date] = X[col_date].astype('datetime64[s]').dt.strftime('%Y-%m-%d %H:%M:%S')
        return dt.Frame(X)

    @staticmethod
    def create_data(X: dt.Frame = None):
        """ Convert transactional data to i.i.d. data by making time-based aggregations """

        if X is None:
            X = TransactionalToIID.make_transactions()
            if not make_features_from_scratch:
                return {'raw_transactions_non_iid': X}

        X_pd = X[:, [col_date, col_group, target]].to_pandas()  # faster, since only working on a few cols
        X_pd[col_row_id] = np.arange(X_pd.shape[0])

        y = X_pd[target]
        y_enc = target + ".enc"

        # Create boolean target
        X_pd[y_enc] = (y == target_labels[1]).astype(int)

        # Make sure time is datetime64, not string
        X_pd[col_date] = pd.to_datetime(X_pd[col_date])

        for leak in leaky_choices:
            # Create the groups
            groups = X_pd.groupby(col_group)

            shift_amount = 0 if leak else 1  # this is critical to avoid leaks!  DO NOT SET IT TO 0 IN PRODUCTION!

            # Compute aggregation over time
            for t in window_length_days:
                t_days = str(t) + "d"  # pandas will do rolling window over this many days ('5d' etc.)
                for op in operators:
                    lag_feature = []
                    for _, df in groups:
                        df = df.sort_values(col_date)
                        time_window = df.set_index(col_date)[y_enc].shift(shift_amount). \
                            fillna(0).rolling(t_days, min_periods=1)  # get time window. if leaky, includes self
                        res = getattr(time_window, op)()  # apply operator on time window
                        res.index = df.index
                        lag_feature.append(res)
                    # Index is set on both side so equal works and reorders rows automatically
                    X_pd["%s%s_%s_past_%d_days_grouped_by_%s" %
                         ("leaky_" if leak else "", op, target, t, col_group)] = pd.concat(lag_feature, axis=0)

        del X_pd[y_enc]  # delete temporary binary response column

        # delete grouping column, since have all aggregations already in iid form
        del X_pd[col_group]
        del X[col_group]

        # create datatable frame of new features (only)
        X_features = dt.Frame(X_pd.loc[:, [x for x in X_pd.columns if x not in [col_date, target, col_row_id]]])

        # add new features to original frame
        X_new = dt.cbind(X, X_features)

        out = {}
        for name, time_range in {
            # 2-way split: ideal for iid, let Driverless do internal validation splits on training split
            'train_iid': X_pd[col_date] <= split_date,
            'test_iid': X_pd[col_date] > split_date
        }.items():
            # X_pd is pandas - easier to deal with time slices, and keep row_id to index into datatable below
            which_rows = X_pd.loc[time_range, col_row_id].reset_index(drop=True).values
            if shuffle:
                np.random.shuffle(which_rows)  # shuffle data for generality - no impact on iid modeling
                name += ".shuf"
            for leak in leaky_choices:
                X_out = X_new.copy()  # shallow copy
                if leak:
                    cols_to_del = [x for x in X_features.names if "leaky" != x[:5]]
                else:
                    cols_to_del = [x for x in X_features.names if "leaky" == x[:5]]
                del X_out[:, cols_to_del]
                out[name + (".leaky" if leak else "")] = X_out[which_rows, :]

        return out


def test_transactional_to_iid():
    ret = TransactionalToIID.create_data()
    for name, X in ret.items():
        le = LabelEncoder()
        y = le.fit_transform(X[target]).ravel()
        print(name)
        print(X.head(10))
        print(X.tail(10))
        for col in X.names:
            if "_past_" in col:
                auc = roc_auc_score(y, X[col].to_numpy().ravel())
                print("%s: auc = %f" % (col, auc))
                if "leaky" not in col:
                    assert auc > 0.53  # all lags must have signal
                    assert auc < 0.8  # but not too much
                else:
                    assert auc > 0.75  # all leaky lags must have a lot of signal
