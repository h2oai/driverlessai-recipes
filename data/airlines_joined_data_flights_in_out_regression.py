"""Create augmented airlines datasets for regression"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class AirlinesData(CustomData):
    # base_url = "http://stat-computing.org/dataexpo/2009/"  # used to work, but 404 now
    base_url = "https://0xdata-public.s3.amazonaws.com/data_recipes_data/"

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import bz2

        def extract_bz2(file, output_file):
            zipfile = bz2.BZ2File(file)
            data = zipfile.read()
            open(output_file, 'wb').write(data)

        temp_path = os.path.join(user_dir(), "recipe_tmp", "airlines")
        os.makedirs(temp_path, exist_ok=True)

        # specify which years are used for training and testing
        training = [2007]
        testing = [2008]

        # download and unzip files
        files = []
        for f in ["%d.csv.bz2" % year for year in training + testing]:
            link = AirlinesData.base_url + "%s" % f
            file = download(link, dest_path=temp_path)
            output_file = file.replace(".bz2", "")
            extract_bz2(file, output_file)
            files.append(output_file)

        # parse with datatable
        X = dt.rbind(*[dt.fread(x) for x in files])

        # add date
        date_col = 'Date'
        X[:, date_col] = dt.f['Year'] * 10000 + dt.f['Month'] * 100 + dt.f['DayofMonth']
        cols_to_keep = ['Date']

        # add number of flights in/out for each airport per given interval
        timeslice_mins = 60
        for name, new_col, col, group in [
            ("out", "CRSDepTime_mod", "CRSDepTime", "Origin"),
            ("in", "CRSArrTime_mod", "CRSArrTime", "Dest")
        ]:
            X[:, new_col] = X[:, dt.f[col] // timeslice_mins]
            group_cols = [date_col, group, new_col]
            new_name = 'flights_%s' % name
            flights = X[:, {new_name: dt.count()}, dt.by(*group_cols)]
            flights.key = group_cols
            cols_to_keep.append(new_name)
            X = X[:, :, dt.join(flights)]

        # Fill NaNs with 0s
        X[dt.isna(dt.f['DepDelay']), 'DepDelay'] = 0
        cols_to_keep.extend([
            'DepDelay',
            'Year',
            'Month',
            'DayofMonth',
            'DayOfWeek',
            'CRSDepTime',
            'UniqueCarrier',
            'FlightNum',
            'TailNum',
            'CRSElapsedTime',
            'Origin',
            'Dest',
            'Distance',
            # Leaks for delay
            # 'DepTime',
            # 'ArrTime', #'CRSArrTime',
            # 'ActualElapsedTime',
            # 'AirTime', #'ArrDelay', #'DepDelay',
            # 'TaxiIn', #'TaxiOut', #'Cancelled', #'CancellationCode', #'Diverted', #'CarrierDelay',
            # #'WeatherDelay', #'NASDelay', #'SecurityDelay', #'LateAircraftDelay',
        ])
        X = X[:, cols_to_keep]

        # Join in some extra info
        join_files = [('UniqueCarrier', 'carriers.csv', 'Code'),
                      ('Origin', 'airports.csv', 'iata'),
                      ('Dest', 'airports.csv', 'iata'),
                      ('TailNum', 'plane-data.csv', 'tailnum')]

        for join_key, file, col in join_files:
            file = download('https://0xdata-public.s3.amazonaws.com/data_recipes_data/%s' % file, dest_path=temp_path)
            X_join = dt.fread(file, fill=True)
            X_join.names = {col: join_key}
            X_join.names = [join_key] + [join_key + "_" + x for x in X_join.names if x != join_key]
            X_join.key = join_key
            X = X[:, :, dt.join(X_join)]
            del X[:, join_key]

        split = False

        if not split:
            filename = os.path.join(temp_path,
                                    "flight_delays_regression_%d-%d.jay" % (min(training), max(testing)))
            X.to_jay(filename)
            return filename
        else:
            # prepare splits (by year) and create binary .jay files for import into Driverless AI
            output_files = []
            for condition, name in [
                ((min(training) <= dt.f['Year']) & (dt.f['Year'] <= max(training)), 'training'),
                ((min(testing) <= dt.f['Year']) & (dt.f['Year'] <= max(testing)), 'test'),
            ]:
                X_split = X[condition, :]
                filename = os.path.join(temp_path, "flight_delays_%s.jay" % name)
                X_split.to_jay(filename)
                output_files.append(filename)
            return output_files
