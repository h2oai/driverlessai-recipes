"""Create hydraulic systems dataset"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

class HydraulicSystemsData(CustomData):
    # url contains hydraulic systems condition monitoring data set
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/"

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import zipfile

        def mean_conversion(pd_frame):
            """
                Means are computed along columns to average the cycle sensor data
                Each row in frame is data collected over a cycle of 60 seconds
            """
            return pd_frame.mean(axis = 1) # axis=1 means along column

        def convert_to_frame(file_bytes_object):
            """
                Convert bytes data to pandas dataframe
            """
            data_file_str = str(file_bytes_object,"utf-8")
            dt_frame = dt.Frame(data_file_str)
            pd_frame = dt_frame.to_pandas()
            return pd_frame

        def import_data_field(zip, filename, fieldname):
            """
                Loads pandas dataframe with average cycle sensor data
            """
            pd_frame = pd.DataFrame( mean_conversion(convert_to_frame(zip.read(filename))) )
            pd_frame.columns = [fieldname]
            return pd_frame

        def import_target_label(zip, filename, target_label):
            """
                Loads pandas dataframe with target label field
            """
            pd_frame = convert_to_frame(zip.read(filename))
            if target_label == "cool_cond_y":
                # store hydraulic cooler condition label data from column 0 into pd frame
                pd_targetLabelY = pd.DataFrame(pd_frame.iloc[:,0])
            elif target_label == "valve_cond_y":
                # store hydraulic valve condition label data from column 1 into pd frame
                pd_targetLabelY = pd.DataFrame(pd_frame.iloc[:,1])
            elif target_label == "pump_leak_y":
                # store hydraulic internal pump leakage label data from column 2 into pd frame
                pd_targetLabelY = pd.DataFrame(pd_frame.iloc[:,2])
            elif target_label == "acc_gas_leak_y":
                # store hydraulic accumulator gas leakage label data from column 3 into pd frame
                pd_targetLabelY = pd.DataFrame(pd_frame.iloc[:,3])
            elif target_label == "stable_y":
                # store hydraulic stable flag label data from column 4 into pd frame
                pd_targetLabelY = pd.DataFrame(pd_frame.iloc[:,4])
            else:
                sys.exit("Not valid hydraulic condition label")
            pd_targetLabelY.columns = [target_label]
            return pd_targetLabelY

        def extract_hydraulic_zip(file, output_file, target_label):
            with zipfile.ZipFile(file,"r") as zip:
                # import all data of pressure sensors
                df_ps1 = import_data_field(zip, "PS1.txt", "psa_bar")
                df_ps2 = import_data_field(zip, "PS2.txt", "psb_bar")
                df_ps3 = import_data_field(zip, "PS3.txt", "psc_bar")        
                df_ps4 = import_data_field(zip, "PS4.txt", "psd_bar")
                df_ps5 = import_data_field(zip, "PS5.txt", "pse_bar")
                df_ps6 = import_data_field(zip, "PS6.txt", "psf_bar")

                # import all data of volume flow sensors
                df_vf1 = import_data_field(zip, "FS1.txt", "fsa_vol_flow")
                df_vf2 = import_data_field(zip, "FS2.txt", "fsb_vol_flow")

                # import all data of temperature sensors
                df_ts1 = import_data_field(zip, "TS1.txt", "tsa_temp")
                df_ts2 = import_data_field(zip, "TS2.txt", "tsb_temp")
                df_ts3 = import_data_field(zip, "TS3.txt", "tsc_temp")
                df_ts4 = import_data_field(zip, "TS4.txt", "tsd_temp")

                # import data of pump efficiency sensor
                df_pe1 = import_data_field(zip, "EPS1.txt", "pump_eff")

                # import data of vibrations sensor
                df_vs1 = import_data_field(zip, "VS1.txt", "vs_vib")

                # import data of cooling efficiency sensor
                df_ce1 = import_data_field(zip, "CE.txt", "cool_eff_pct")

                # import data of cooling power sensor
                df_cp1 = import_data_field(zip, "CP.txt", "cool_pwr_pct")

                # import data of efficiency factor sensor
                df_ef1 = import_data_field(zip, "SE.txt", "eff_fact_pct")

                # import one of the hydraulic condition monitoring labels from profile based on target label
                df_hydCondMonitorLabelY = import_target_label(zip, "profile.txt", target_label)        
                
                # Combine all sensor Dataframes and then add hydraulic condition monitoring label as last column
                df_hydCondMonitorDataX = pd.concat([df_ps1, df_ps2, df_ps3, df_ps4, df_ps5, df_ps6, df_vf1, \
                                                df_vf2, df_ts1, df_ts2, df_ts3, df_ts4, df_pe1, df_vs1, \
                                                df_ce1, df_cp1, df_ef1, df_hydCondMonitorLabelY], axis=1)
                dt.Frame(df_hydCondMonitorDataX).to_csv(output_file) 
                print(df_hydCondMonitorDataX.columns)

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "hydraulic")
        os.makedirs(temp_path, exist_ok=True)

        link = HydraulicSystemsData.base_url + "data.zip"
        file = download(link, dest_path=temp_path)
        # Choose a target_label: cool_cond_y, valve_cond_y, pump_leak_y, acc_gas_leak_y, stable_y
        # The target_label you choose will get added on as the last column in the training dataset
        target_label = "cool_cond_y"
        output_file = file.replace("data.zip", "hydCondMonitorData.csv")
        print("%s %s" % (file, output_file))        
        extract_hydraulic_zip(file, output_file, target_label)

        return output_file