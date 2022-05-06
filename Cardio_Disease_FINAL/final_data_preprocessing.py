import pandas as pd

df = pd.read_csv('processed_data.csv')

new_df = df.iloc[:,8]
new_df = new_df.to_frame()

index_to_be_removed = []

for i in new_df.index:
    
    if(new_df['ap_lo'][i] >= 200 or new_df['ap_lo'][i] <= 0):
        index_to_be_removed.append(i)
        

df = df.drop(index_to_be_removed)

df.to_csv('final_data.csv')