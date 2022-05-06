import pandas as pd
import random

df = pd.read_csv('data.csv')

new_df = df.iloc[:,-1]
new_df = new_df.to_frame()


index_of_1 = []
index_of_0 = []

count = 0
count2 = 0

for ind in new_df.index:
    
    if(new_df['cardio'][ind] == 1):
        index_of_1.append(ind)
        count = count + 1
        
    elif(new_df['cardio'][ind] == 0):
        index_of_0.append(ind)
        count2 = count2 + 1

for i in range(2000):
    
    index_of_1.remove(random.choice(index_of_1))
    
for i in range(2000):
    
    index_of_0.remove(random.choice(index_of_0))
    
df = df.drop(index_of_1)
df = df.drop(index_of_0)

df = df.round({'age_year' : 0})

df.to_csv('processed_data.csv')