#%%
# Disable warning
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#%%
import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

DS_PATH=os.getenv('DS_PATH','')

df = pd.read_csv(DS_PATH, header=1, sep=",", index_col=0, parse_dates=True, error_bad_lines=False)

#%%
df.shape


#%%
df.head(5)


#%%
df.tail(5)