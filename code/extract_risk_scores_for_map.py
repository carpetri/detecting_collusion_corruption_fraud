# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

contracts = pd.read_csv('../Data/Procurements/awards_with_features.csv', index_col=0)

# <codecell>

contracts1 = pd.read_csv('../Data/Procurements/Historic_and_Major_awards.csv', index_col=0)

# <codecell>

networks = pd.read_csv('../Data/Procurements/Historic_and_Major_awards_network_features.csv', index_col=0)

# <codecell>

aggregations = {
    'unique_id': 'count'
}
grouped = contracts.groupby(by='buyer_country', as_index=False).agg(aggregations)

# <codecell>

for c in grouped['buyer_country'].unique(): print c

# <codecell>

grouped.to_csv('../Data/Procurements/contracts_per_country.csv')

# <codecell>

countries = pd.read_csv('../../worldbank/Data/Procurements/countries_with_points.csv', header=None)

# <codecell>

contracts_short = contracts[contracts['buyer_country'].isin(countries[1])].sort(columns='buyer_country')

# <codecell>

contracts_short.index = range(0,len(contracts_short))

# <codecell>

contract_ids = contracts_short[['buyer_country','unique_id']]

# <codecell>

risk = pd.read_csv('../../worldbank/Data/contract_risk.csv', index_col=0)

# <codecell>

risk.columns

# <codecell>

contract_ids = pd.merge(contract_ids, risk, left_on='unique_id', right_on='unique_id_contracts', how='left')

# <codecell>

contract_ids.shape

# <codecell>

contract_ids.to_csv('../../worldbank/Data/Procurements/contract_ids.csv')

