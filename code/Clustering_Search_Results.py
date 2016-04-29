# -*- coding: utf-8 -*-
import pandas as pd
from numpy import intersect1d
from scipy import sparse
import networkx as nx
import csv

# Load Data, Sort It, and Drop Header Rows from CSV File Format
queries = pd.read_csv('../Search_Results/all_entities_Google_results.csv', 
    index_col=0)
queries = queries.sort_index()
queries = queries.drop(queries.index[where(queries["Name"]=="Name")[0]])
queries.to_csv('../Search_Results/all_entities_Google_results.csv',
 quoting=csv.QUOTE_ALL)
query_results = queries.values[:,1:]

# Define Comparison and Linking Function
def compare_ij(i,j):
    return [y for y in ##All the elements
            set(query_results[i]).intersection(query_results[j]) ###That are in both sets of query results
            if type(y)==str] ###That are strings (i.e. Not nans)
#     Use the below code if you wanted to keep track of rankings. It's slower by up to a factor of 3 in the worst case.
#     dict1 = {}
#     dict2 = {}
#     list1 = query_results[i]
#     list2 = query_results[j]
#     indices = xrange(10)
#     for ele in indices:
#         dict1[list1[ele]] = ele
#     for ele in indices:
#         value = list2[ele]
#         if value in dict1:
#             dict2[value] = ele
#     intersect = [y for y in dict2.keys() if type(y)==str]
#     ranks = [(dict2[value],dict1[value]) for value in intersect]
#     return intersect, ranks

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
    
    entity_i = clean_string_ends(entity_i)    
    
    for j in arange(i+1, n_entities):
        entity_j = queries.ix[j,"Name"]
        
        entity_j = clean_string_ends(entity_j)

        matches = compare_ij(i,j)
        attr_dict = {"matches": matches,
                     "n_matches": len(matches),
                     }
        
        if matches:
            match_graph.add_edge(entity_i, entity_j, attr_dict=attr_dict)   
    if match_graph.number_of_nodes():
        fh=codecs.open( '../Search_Clustering/%i_matches.el' %i, mode='w', encoding='utf-8')
        for line in generate_edgelist(match_graph, delimiter, data=True):
            line+='\n'
            fh.write(line)
        fh.close()


from multiprocessing import Pool
p = Pool()

%%prun
p.map(find_links, arange(n_entities))