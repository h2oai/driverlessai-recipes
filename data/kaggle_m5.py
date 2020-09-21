"""Prepare data for m5 Kaggle Time-Series Forecast competition"""
import os
import uuid
from collections import OrderedDict
from zipfile import ZipFile

from h2oaicore.data import CustomData

import pandas as pd
import datatable as dt

from h2oaicore.systemutils import user_dir
from h2oaicore.systemutils_more import download

tmp_dir = os.path.join(user_dir(), str(uuid.uuid4())[:6])
path_to_zip = "https://files.slack.com/files-pri/T0329MHH6-F0150BK8L01/download/m5-forecasting-accuracy.zip?pub_secret=acfcbf3386"

holdout_splits = {
    'm5_private': range(1942, 1942 + 28)  # private LB
}


class PrepareM5Data(CustomData):
    """ Prepare data for m5 Kaggle Time-Series Forecast competition"""

    @staticmethod
    def create_data(X: dt.Frame = None):
        file = download(url=path_to_zip, dest_path=tmp_dir)
        with ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        num_id_cols = 6
        main_data = dt.fread(os.path.join(tmp_dir, "sales_train_evaluation.csv"))
        all_cols = list(main_data.names)
        id_cols = all_cols[:num_id_cols]
        date_cols = all_cols[num_id_cols + 1125:]

        # training data
        target = "target"
        data = pd.melt(main_data.to_pandas(), id_vars=id_cols, value_vars=date_cols, var_name="d", value_name=target)
        data[target] = data[target].astype(float)
        data = dt.Frame(data)
        data_splits = [data]
        names = ["m5_train"]

        # test data for submission
        submission = dt.fread(os.path.join(tmp_dir, "sample_submission.csv"))

        for name, ranges in holdout_splits.items():
            test_cls = ["d_" + str(k) for k in ranges]
            test_data = []
            ids = submission["id"].to_list()[0]
            new_test_cols = ["d"] + id_cols
            for i in range(len(ids)):
                id = ids[i]
                splits = ids[i].split("_")
                item_id = splits[0] + "_" + splits[1] + "_" + splits[2]
                dept_id = splits[0] + "_" + splits[1]
                cat_id = splits[0]
                store_id = splits[3] + "_" + splits[4]
                state_id = splits[3]
                id_values = [id, item_id, dept_id, cat_id, store_id, state_id]
                for j in range(len(test_cls)):
                    row_values = [test_cls[j]] + id_values
                    test_data.append(row_values)

            test_data = pd.DataFrame(test_data, columns=new_test_cols)
            test_data = dt.Frame(test_data)
            data_splits.append(test_data)
            names.append(name)

        weather_data = dt.fread(os.path.join(tmp_dir, "calendar.csv"))
        weather_data.key = "d"

        price_data = dt.fread(os.path.join(tmp_dir, "sell_prices.csv"))
        price_data.key = ["store_id", "item_id", "wm_yr_wk"]

        ret = OrderedDict()
        for n, f in zip(names, data_splits):
            f = f[:, :, dt.join(weather_data)]
            f = f[:, :, dt.join(price_data)]
            ret[n] = f
        return ret
