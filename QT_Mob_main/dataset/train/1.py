import pandas as pd
df=pd.read_feather("QT_Mob_main/dataset/test/inner_data_seq_dataset.feather")
df.loc[0].to_csv("QT_Mob_main/dataset/test/inner_data_seq_dataset_sample.csv",index=False)
print(df.loc[0,'user'])
print(df.loc[0,'response'])
print(df.loc[0,'prediction'])
print(df.loc[0,'inters'])
print(df.loc[0,'time'])
print(df.loc[0,'profile'])