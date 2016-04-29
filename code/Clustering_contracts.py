# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Clustering unlabeled contracts
# - Choose features for clustering
# - Deal with NaNs
# - Choose clustering algorithm (k-means vs. mean shift)
# - choose number of clusters if necessary
# - summarize clusters

# <codecell>

import pandas as pd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
from random import sample
%matplotlib inline

# <markdowncell>

# ###Load data set and select features for clustering
#  - To perform clustering, edit filepath in cell below to point to data set
#  - Edit list of features in cell below to use in clustering

# <codecell>

#df_full = pd.read_hdf('../Data/MDB_data/WorldBank/world_bank_combined_clean.hdf5','df')
df_full = pd.read_csv('../../worldbank/Data/Procurements/MDB_data/WorldBank/Historic_and_Major_awards.csv',index_col=0)
for col in df_full.columns:
    print col, (float(sum(df_full[col].isnull())) / df_full.shape[0])

# <codecell>

#choose columns from data frame to use as features
columns = [
    'award_amount_usd', 
    'competitive',
    'number_of_bids',
    'total_award_amount_C/Y', 
    'total_contracts_C/Y',
    'total_award_amount_C/S/Y', 
    'total_contracts_C/S/Y',
    'gini_index_mean',
    'unemployment_perc_mean',
    'gdp_per_capita'
]

df = df_full[['buyer_country', 'canonical_name'] + columns]

# <codecell>

df.shape

# <markdowncell>

# ###Remove/replace NaNs in features
# - Haven't implemented ... probably data set & feature specific

# <codecell>

# Attach investigations data
investigations = pd.read_csv('../Data/Investigations/investigations.csv', index_col=0)
investigations = investigations[['unique_id','canonical_name', 'country']].drop_duplicates(['canonical_name', 'country'])
df = pd.merge(df, investigations, left_on=['canonical_name', 'buyer_country'], \
              right_on=['canonical_name', 'country'], how='left')

# <codecell>

#Deal with NaNs as needed
df = df.dropna(subset=[columns])

# <markdowncell>

# ##Set number of clusters for k-means
# For the k-means clustering algorithm, the number of clusters k must be chosen.
# K can be input manually, calculated using a general rule-of-thumb (sqrt(n/2)), or optimized according to silhouette scores and percent of variance explained by various values of k.
# 
# ### How to interpret charts
# 
# The first chart shows percent of variance explained by clustering at different values of k. This chart is often used as a visual aid to choosing k, where an "elbow" in the curve indicates an appropriate k value. This chart is less helpful when there is no obvious elbow.
# 
# The second chart shows silhouette scores at different values of k which is a measure of how well each data point fits within its cluster. A peak in this curve corresponds to the best value of k according to the silhouette measure.

# <codecell>

#Create numpy array from dataframe
df_sample = df.ix[sample(df.index, 10000)]
Z = df_sample[columns].values
X = df[columns].values

# <codecell>

#k = 10
#k = math.sqrt(X.shape[0] / 2)

#Choose based on % variance explained & silhouette scores
k_max = 10 #Set max number of clusters
k_range = np.arange(2,k_max+1)
TSS = KMeans(n_clusters=1).fit(Z).inertia_
silhouette_score = np.empty([k_max-1,])
var_explained = np.empty([k_max-1,])

for k in k_range:
    kmeans_model = KMeans(n_clusters=k).fit(Z)
    labels = kmeans_model.labels_
    silhouette_score[k-2] = metrics.silhouette_score(Z, labels, metric='euclidean')
    var_explained[k-2] = (TSS - kmeans_model.inertia_) / TSS

fig1 = plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace = .1)
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212, sharex=ax1)
ax1.plot(k_range, var_explained, linewidth=2)
ax2.plot(k_range, silhouette_score, linewidth=2)
setp(ax1.get_xticklabels(), visible=False)
ax1.set_ylabel('% of variance explained')
ax2.set_ylabel('Silhouette Score')
ax2.set_xlabel('Number of clusters')

# <codecell>

#Choose k based on charts above
k = 5
model = KMeans(n_clusters=k).fit(X)
labels = model.labels_
df['cluster'] = labels

# <codecell>

# mean-shift clustering algorithm. comment out if using kmeans ...

# bandwidth = estimate_bandwidth(Z, quantile=0.1, n_samples=1000)
# model = MeanShift(bandwidth=bandwidth).fit(Z)
# labels = model.labels_
# df_sample['cluster'] = labels

# <markdowncell>

# ##Summarize clusters
# Summarize clusters across features for each cluster.

# <codecell>

#Create box-and-whisker plots by cluster and feature
grouped = df.groupby(by='cluster', as_index=False)
clusters = grouped['award_amount_usd'].agg('count')
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.bar(clusters['cluster'], clusters['award_amount_usd'], align='center')
plt.xlabel('cluster')
plt.ylabel('percent overlap with investigations')


def perc_na(group):
    return (1 - (sum(group.isnull())/group.shape[0]))*100
inv = grouped['unique_id'].agg(perc_na)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.bar(inv['cluster'], inv['unique_id'], align='center')
plt.xlabel('cluster')
plt.ylabel('percent overlap with investigations')

for field in columns:
    data = []
    clusters = []
    for clusterID in df['cluster'].unique():
        data.append([df[field][df['cluster']==clusterID].values])
        clusters.append(clusterID)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data, positions = clusters)
    
    means = [np.mean(x) for x in data]
    plt.scatter(clusters, means)

    plt.xlabel('Cluster')
    plt.ylabel(field)
    plt.tight_layout()

# <codecell>

def perc_na(group):
    return (1 - (sum(group.isnull())/group.shape[0]))*100

inv = grouped['unique_id'].agg(perc_na)
inv

# <codecell>

#Create box-and-whisker plots by cluster and feature
grouped = df_sample.groupby(by='cluster', as_index=False)
clusters = grouped['award_amount_usd'].agg('count')
plt.bar(clusters.index, clusters['award_amount_usd'], align='center')

for field in columns:
    data = []
    clusters = []
    for clusterID in df_sample['cluster'].unique():
        data.append([df_sample[field][df_sample['cluster']==clusterID].values])
        clusters.append(clusterID)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data, positions = clusters)
    
    means = [np.mean(x) for x in data]
    plt.scatter(clusters, means)

    plt.xlabel('Cluster')
    plt.ylabel(field)
    plt.tight_layout()

# <codecell>

#Create graph of feature means (bars), medians (horiz. lines), and standard deviations (error bars)
grouped = df.groupby('cluster')
means = grouped.agg('mean')
stds = grouped.agg('std')
medians = grouped.agg('median')

for field in means.columns:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.bar(means.index, means[field], yerr=stds[field], 
           align='center',
           color='r',
           ecolor='k',
           alpha=0.4)
    ax.plot(medians[field], 'k_', markersize=50, markeredgewidth=1.5)

    plt.xlabel('Cluster')
    plt.ylabel(field)
    plt.tight_layout()

