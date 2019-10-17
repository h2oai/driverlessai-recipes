"""Create some non-trivial airlines datasets"""

from h2oaicore.data import CustomData

## AIRLINE DATA - END-TO-END DATA PREPARATION

##  1) Download selected yearly airline datasets from http://stat-computing.org/dataexpo/2009/
##  2) Unzip all .bz2 files
##  3) Concatenate all files
##  4) Select flights leaving from SFO only
##  5) Create a linear date (time) column for time-series modeling
##  6) Compute the number of scheduled flights in/out-bound for a given airport, for each hour
##  7) Create a binary target column (Departure Delay > 15 minutes)
##  8) Join Carrier, Airport and Plane data, also downloaded from http://stat-computing.org/dataexpo/2009/
##  9) Optionally: Split the data by time
##  10) Import the data into Driverless AI for further experimentation


class AirlinesData(CustomData):
    def create_data(data: dt.Frame = None):
        import datatable as dt
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import bz2

        def extract_bz2(file, output_file):
            zipfile = bz2.BZ2File(file)
            data = zipfile.read()
            open(output_file, 'wb').write(data)

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "airlines")
        os.makedirs(temp_path, exist_ok=True)
        dt.options.nthreads = 8

        # specify which years are used for training and testing
        training = list(range(2005, 2008))
        testing = [2008]

        # download and unzip files
        files = []
        for f in ["%d.csv.bz2" % year for year in training + testing]:
            link = "http://stat-computing.org/dataexpo/2009/%s" % f
            file = download(link, dest_path=temp_path)
            output_file = file.replace(".bz2", "")
            if not os.path.exists(output_file):
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
            new_name = 'flights_%s_per_%d_min' % (name, timeslice_mins)
            flights = X[:, {new_name: dt.count()}, dt.by(*group_cols)]
            flights.key = group_cols
            cols_to_keep.append(new_name)
            X = X[:, :, dt.join(flights)]

        # select flights leaving from SFO only
        X = X[dt.f['Origin'] == 'SFO', :]

        # Fill NaNs in DepDelay column
        X[dt.isna(dt.f['DepDelay']), 'DepDelay'] = 0

        # create binary target column
        depdelay_threshold_mins = 15
        target = 'DepDelay%dm' % depdelay_threshold_mins
        X[:, target] = dt.f['DepDelay'] > depdelay_threshold_mins
        cols_to_keep.extend([
            target,
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
            file = download('http://stat-computing.org/dataexpo/2009/%s' % file, dest_path=temp_path)
            X_join = dt.fread(file, fill=True)
            X_join.names = {col: join_key}
            X_join.names = [join_key] + [join_key + "_" + x for x in X_join.names if x != join_key]
            X_join.key = join_key
            X = X[:, :, dt.join(X_join)]
            del X[:, join_key]

        split = True
        if not split:
            filename = os.path.join(temp_path,
                                    "flight_delays_data_recipe_%d-%d.csv" % (min(training), max(testing)))
            X.to_csv(filename)
            return filename
        else:
            # prepare splits (by year) and create binary .jay files for import into Driverless AI
            output_files = []
            for condition, name in [
                ((min(training) <= dt.f['Year']) & (dt.f['Year'] <= max(training)), 'training'),
                ((min(testing) <= dt.f['Year']) & (dt.f['Year'] <= max(testing)), 'test'),
            ]:
                X_split = X[condition, :]
                filename = os.path.join(temp_path, "augmented_flights_%s-%d_%s.csv" %
                                        (X_split[:, 'Year'].min1(), X_split[:, 'Year'].max1(), name))
                X_split.to_csv(filename)
                output_files.append(filename)
            return output_files
