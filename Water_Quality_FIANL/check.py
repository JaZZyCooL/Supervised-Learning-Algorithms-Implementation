import pandas as pd

df = pd.read_csv('final_data.csv')

new_df = df.iloc[:,-1]
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
