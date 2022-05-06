import pandas as pd
import random

df = pd.read_csv('water_potability.csv')

df = df.dropna(axis = 0, how='any', thresh=None, subset=None, inplace=False)

new_df = df.iloc[:, -1]
new_df = new_df.to_frame()

index_of_1 = []
index_of_0 = []

count = 0
count2 = 0

for ind in new_df.index:
    
    if(new_df['Potability'][ind] == 1):
        
        index_of_1.append(ind)
        count = count + 1
        
    elif(new_df['Potability'][ind] == 0):
        
        index_of_0.append(ind)
        count2 = count2 + 1
        
index_to_be_removed_0 = []

for i in range(400):
    
    index_to_be_removed_0.append(random.choice(index_of_0))

df = df.drop(index_to_be_removed_0)

df.to_csv('final_data.csv')