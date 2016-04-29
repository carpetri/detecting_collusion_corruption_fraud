# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Creates a list of all company and entity names across all data from World Bank, African Development Bank, and country data sources. These entities are include suppliers that have bid on procurement contracts and all entities that are debarred or went through investigation by the world bank. 

# <codecell>

data_list = [
'../data_clean/Historic_and_Major_awards.csv',
'../Data/Procurements/MDB_data/AfricanDB/AfDB_clean.csv',
'../Data/Procurements/MDB_data/AsianDB/asianDB_data_clean/asian_db_clean.csv',
'../Data/Procurements/country_data/Mexico/mex_clean.csv',
'../Data/Procurements/country_data/Tanzania/tanzania_clean.csv',
'../Data/Procurements/country_data/Kenya/kenya_clean.csv',
'../Data/Procurements/country_data/Senegal/senegal_clean.csv',
'../Data/Procurements/country_data/Liberia/Liberia.csv',
'../Data/Procurements/country_data/Uganda/uganda_clean.csv',
'../Data/Procurements/country_data/Madagascar/madagascar.csv',
'../Data/Procurements/country_data/Nepal/nepal.csv',
'../Data/Procurements/country_data/Vietnam/vietnam.csv',
'../Data/Debarment/world_bank_debarred_companies.csv', # <--- WB debarred
'../Data/Procurements/Toms Data -- companies.csv', # <--- Tom's data
'../Data/Investigations/libre-office-encoded-CMS-Management-Report-20140626-103453.csv'
]

import pandas as pd
import numpy as np
all_entities = pd.Series()

for data_file in data_list:
    data = pd.read_csv(data_file,dtype=str)
    if 'supplier' in data.columns:
        entities = data['supplier']
    elif 'bidder.name' in data.columns:
        entities = data['bidder.name'] #Tom's Data
    elif 'Subject' in data.columns:
        entities = data['Subject'] #Investigation data
    else:
        raise Exception("I don't know how to handle this data: %s"%data_file)

    #print data_file
    #print where(entities.map(f))
    if len(all_entities) == 0:
        all_entities = entities
    else: 
        all_entities = all_entities.append(entities)

all_entities = all_entities.drop_duplicates()
all_entities.index = np.arange(all_entities.shape[0])
all_entities = pd.DataFrame(data=all_entities.values, columns=['Name'])

# <markdowncell>

# Shuffle items so we can't infer from position which entities were investigation data vs. not

# <codecell>

from numpy import random
def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(random.shuffle, axis=axis)
    return df
all_entities = shuffle(all_entities)

# <markdowncell>

# Write to disk.

# <codecell>

import csv
all_entities.to_csv('../data_clean/all_entities.csv', quoting=csv.QUOTE_ALL)
# all_entities.to_hdf('../Data/Entities/all_entities.h5', 'df')

