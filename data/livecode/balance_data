# Create a sample dataset for imbalanced use cases - not for modeling but can be nice to better see trends in MLI PDP plots

from sklearn.utils import resample
import pandas as pd

target_column = "Known_Fraud"

df = X.to_pandas()

df_majority = df[df[target_column]==0]
df_minority = df[df[target_column]==1]

g = df.groupby(target_column)
n = g.size().min()
 
df_majority_downsampled = resample(df_majority, 
                                 replace=False,     
                                 n_samples=n*5,     
                                 random_state=123)  
 
return pd.concat([df_majority_downsampled, df_minority])

