# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Setup
# -----

# <codecell>
from pylab import *
import pandas as pd
from numpy import intersect1d
from scipy import sparse
import networkx as nx

# <markdowncell>

# Load Data, Sort It, and Drop Header Rows from CSV File Format
# ----

# <codecell>

queries = pd.read_csv('all_entities_Google_results.csv', index_col=0)
queries = queries.sort_index()
queries = queries.drop(queries.index[where(queries["Name"]=="Name")[0]])
import csv
queries.to_csv('all_entities_Google_results.csv', quoting=csv.QUOTE_ALL)

query_results = queries.values[:,1:]

# <markdowncell>

# Get Top Level Domains and Frequencies
# ----

# <codecell>

# def get_tld(result):
#     if type(result) is float and isnan(result):
#         return result
#     else:
#         return result.split('/')[2]

# <codecell>

# tld_query_results = query_results.copy()

# tld_query_results[map(str, range(10))] = tld_query_results[map(str, range(10))].applymap(get_tld)

# <codecell>

# term_frequencies = pd.Series(tld_query_results[map(str, range(10))].values.ravel()).value_counts()

# <markdowncell>

# Define Comparison and Linking Function
# ----

# <codecell>

def compare_ij(i,j):
    return [y for y in ##All the elements
            set(query_results[i]).intersection(query_results[j]) ###That are in both sets of query results
            if type(y)==str] ###That are strings (i.e. Not nans)

n_entities = len(query_results)
import codecs
generate_edgelist = nx.readwrite.edgelist.generate_edgelist
delimiter=";;;"

def clean_string_ends(s,forbidden_character=';'):
    while s.startswith(';'):
        s = s[1:]
    while s.endswith(';'):
        s = s[:-1]
    return s

def find_links(i):
    match_graph = nx.Graph()
    entity_i = queries.ix[i,"Name"]
    print i, entity_i

    if type(entity_i)!=str: #if it's nan, don't analyze it
        return
    
    entity_i = clean_string_ends(entity_i)    
    
    for j in arange(i+1, n_entities):
        entity_j = queries.ix[j,"Name"]
	if type(entity_j)!=str: #if it's nan, don't analyze it
            continue
        
        entity_j = clean_string_ends(entity_j)

        matches = compare_ij(i,j)
        attr_dict = {"matches": matches,
                     "n_matches": len(matches),
                     }
        
        if matches:
            match_graph.add_edge(entity_i, entity_j, attr_dict=attr_dict)   
    fh=codecs.open('%i_matches.el'%i,mode='w')#,encoding='utf-8')
    if match_graph.number_of_nodes():
        for line in generate_edgelist(match_graph, delimiter, data=True):
            line+='\n'
            fh.write(line)
    fh.close()

# <codecell>

from multiprocessing import Pool
p = Pool()

# <codecell>

p.map(find_links, arange(n_entities))

