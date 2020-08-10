"""Manually add features based on the average target value for similar events to a dataset"""

"""

This recipe adds the average value of the target for recent similar events (where similar events have the same 
values for the categorical variables on the event list).

Settings for Driverless AI:
1. Update folder_path to the data file and the filename.
2. Edit the seconds ahead list so that it lists the number of seconds ahead of time,
that predictions must be made.  A separate file, with separate predict ahead intervals will be 
created for each value on the list. For instance [24*3600, 7*24*3600] would create separate files
with day ahead and week ahead features.
3. Specify the target column.
4. Specify the datetime column.
5. Specify the columns used to define similar events as events.
6. Specify the time intervals over which events will be averaged in seconds as the event_intervals.
eg [1*24*3600, 3*24*3600, 7*24*3600] creates event features averaged over 1, 3, and 7 days.
7. Minimum number of event categories to consider in creating the lagged features. If n=2, all combinations of 2, 3, ... N events from the 
events list are used to define similar events when creating features.
8. Upload under 'ADD DATASET' -> 'UPLOAD DATA RECIPE'
"""

import datatable as dt
import numpy as np
import os

from h2oaicore.data import CustomData
from h2oaicore.systemutils import config


class MyData(CustomData):
    
    @staticmethod
    def create_data():
        
        _modules_needed_by_name = ['datetime']        
        
        import datetime
        import pandas as pd
        from collections import defaultdict
        from itertools import combinations
        
        """
        Update the below as needed
        """
        # Path to the data
        folder_path = 'tmp/'  
        # Data file
        data_file = 'OTG_data_with_datetime.csv' # Data file

        # Number of seconds ahead that predictions should be made
        seconds_ahead_list = [2*24*3600]
        # Target column
        target = "Meals Served"
        # Datetime column
        datetime_column = "datetime"
        # Event group columns
        events = ['Meal Period', 'Concept/Truck', 'Service Location', 'Menu Item Name']
        # time period over which to average events
        event_intervals = [1*24*3600, 3*24*3600, 7*24*3600]

        # minimum number of events to include in combinations
        min_event_combo_number = max(len(events) - 1, 1)
        
        # Try to calculate a datetime
        def create_datetime(x):
            
            try:
                answer = pd.to_datetime(str(x))
            except:
                answer = x
        
            return answer

        
        # Create datasets with minimum features calculated the given number of days ahead
        dataset_dict = {}
        for seconds_ahead in seconds_ahead_list:
            
            train = pd.read_csv(os.path.join(folder_path, data_file))

            # Change the beginning and end of service times to datetimes
            train['datetime'] = train[datetime_column].apply(create_datetime)
                        
            # Calculate all combinations of the even columns that will be used to define a similar event
            event_combinations = []
            for num_in_set in range(min_event_combo_number, len(events) + 1):
                    event_combinations += list(combinations(events, num_in_set))
                    
            for event_categories in event_combinations:
                
                event_categories = list(event_categories)
                
                event_prefix = "previous_"
                for item in event_categories:
                    event_prefix += str(item.replace(' ', '')) + '_'
                    
                temp_shift = train.copy()
                
                # Save separate dataframes for each unique event type
                unique_categories = temp_shift[event_categories].drop_duplicates()   
                
                split_set = {}
    
                # Split the training set by category
                for ii in range(len(unique_categories)):
                    AA = temp_shift.copy()
                    for jj in range(len(event_categories)):
                        AA = AA[AA[event_categories[jj]] == unique_categories.iloc[ii, jj]]
                    split_set[tuple(unique_categories.iloc[ii,:])] = AA
    
                def mean(x):
                    x = list(x)
                    try:
                        answer = sum(x) / float(len(x))
                    except:
                        answer = np.nan
                    return answer
                
                def most_recent(row, seconds_ahead, event_interval, event_categories):
                    # Find the average target value over the given event interval
                    try:
                        train_category = split_set[tuple(row[event_categories])].copy()
                    
                        train_category = train_category[((row['datetime'] - train_category['datetime']).apply(lambda x: x.total_seconds()) >= seconds_ahead) & 
                                                        ((row['datetime'] - train_category['datetime']).apply(lambda x: x.total_seconds()) <= seconds_ahead + event_interval)]
                        answer = mean(train_category[target])
        
                    except:
                        answer = np.nan
                        
                    return answer
    
                # Average recent events over each interval length
                for average_interval in event_intervals:
                    
                    temp_shift[event_prefix + 'event_ave_' + str(average_interval)] = temp_shift.apply(lambda row: most_recent(row, seconds_ahead, average_interval, event_categories), axis=1)
    

                train = temp_shift.copy()

              
            # Save the dataset corresponding to the number of seconds ahead the predictions are being made
            new_name = data_file.split('.')[0] + '_' + str(min_event_combo_number)  + '_event_lags_'
            dataset_dict[new_name + str(seconds_ahead)] = train

            
        return dataset_dict