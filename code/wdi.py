#!/usr/bin/python
# -*- coding: utf-8 -*-

# Obtain Country/Sector Specific Features For Integration With Procurements

import pandas as pd
import numpy as np
import wbdata

indicators = ['IC.BUS.DISC.XQ', 'IC.FRM.CMPU.ZS', 'IC.FRM.CORR.ZS', 'IC.FRM.INFM.ZS', 'IC.LGL.CRED.XQ',
              'IC.LGL.DURS', 'IC.TAX.GIFT.ZS', 'IQ.CPA.PROP.XQ', 'IQ.CPA.TRAN.XQ', 'NY.GDP.PCAP.CD', 
              'SE.PRM.PRSL.ZS', 'SI.POV.GINI', 'SL.UEM.TOTL.NE.ZS']
yearsToKeep = range(1990,2014)
df_procurements = pd.read_csv('../data_clean/Historic_and_Major_awards.csv')

#Clean country names
# 'clean_country_names.csv' has cleaned names from procurements and investigations data
country_names_map = pd.read_csv('../data_clean/clean_country_names.csv')





# country_names_map = country_names_map.append(pd.DataFrame({'dirty':list(additional_dirty_names), 'clean':nan}), ignore_index=True)
# country_names_map = country_names_map.drop_duplicates() # is this necessary?.. i may've appended twice before by accident
# country_names_map.sort(columns='dirty').to_string().split('\n')

# <codecell>

dictForCountryRenaming = {
'American Samoa':'Samoa',
'Andean Countries':'Andean Region',
# 'Andorra':'Andorra',
# 'Arab World':'Arab World',
# 'Aruba':'Aruba',
'Bahamas, The':'Bahamas',
'Brunei Darussalam':'Brunei',
'Cabo Verde':'Cape Verde',
'Caribbean small states':'Caribbean', #???
'Congo, Dem. Rep.':'Congo Democratic Republic',
'Congo, Rep.':'Congo Republic',
#'East Asia & Pacific (all income levels)':'East Asia and Pacific',
#'East Asia & Pacific (developing only)':'East Asia and Pacific',
'Egypt, Arab Rep.':'Egypt',
'Hong Kong SAR, China':'Hong Kong',
'Iran, Islamic Rep.':'Iran',
'Korea, Dem. Rep.':'Korea Democratic Republic',
'Korea, Rep.':'Korea Republic',
#'Latin America & Caribbean (all income levels)':'Latin America and the Caribbean',
#'Latin America & Caribbean (developing only)':'Latin America and the Caribbean',
#'Latin America and the Caribbean (IFC classific...':'Latin America and the Caribbean',
'Macao SAR, China':'Macao',
'Marshall Island':'Mauritania', # mistake Carlos made (?), so I'm fixing it here
'Macedonia, FYR':'Macedonia',
'Micronesia, Fed. Sts.':'Micronesia',
'Not classified':nan,
'Other':nan,
#'Other small states':nan,
#'Sub-Saharan Africa (IFC classification)':'Sub-Saharan Africa',
#'Sub-Saharan Africa (all income levels)':'Sub-Saharan Africa',
#'Sub-Saharan Africa (developing only)':'Sub-Saharan Africa'
}

# <codecell>

country_names_map.clean.fillna(country_names_map.dirty, inplace=True)
country_names_map.clean.replace(dictForCountryRenaming, inplace=True)
country_names_map = country_names_map.drop_duplicates()
country_names_map
country_names_map.to_csv('../data_clean/clean_country_names_2.csv',quoting=csv.QUOTE_ALL)

# <markdowncell>

# #Reshape country indicators for integration with other data

# <codecell>

# first, use the previous map of country names to rename countries in the dfCS data
dfCS['Country Name'].replace(country_names_map.dirty.tolist(), country_names_map.clean.tolist(), inplace=True)
dfCS = dfCS[dfCS['Country Name'].notnull()]

# <codecell>

indicators = ['IC.BUS.DISC.XQ', 'IC.FRM.CMPU.ZS', 'IC.FRM.CORR.ZS', 'IC.FRM.INFM.ZS', 'IC.LGL.CRED.XQ',
              'IC.LGL.DURS', 'IC.TAX.GIFT.ZS', 'IQ.CPA.PROP.XQ', 'IQ.CPA.TRAN.XQ', 'NY.GDP.PCAP.CD', 
              'SE.PRM.PRSL.ZS', 'SI.POV.GINI', 'SL.UEM.TOTL.NE.ZS']
index = pd.MultiIndex.from_product(iterables=[dfCS['Country Name'].unique(), yearsToKeep], names=['country', 'year'])
dfReshaped = pd.DataFrame(columns=indicators, index=index)
dfReshaped

# <codecell>

minYear = str(min(yearsToKeep))
for ctry in dfCS['Country Name'].unique():
    for indic in indicators:
        try: dfReshaped[indic][ctry] = dfCS[(dfCS['Country Name']==ctry) & (dfCS['Indicator Code']==indic)].loc[:,minYear:].T
        except:
            print ctry, indic
            print dfReshaped[indic][ctry]
            print dfCS[(dfCS['Country Name']==ctry) & (dfCS['Indicator Code']==indic)].loc[:,minYear:].T
            raise
            

# <codecell>

dfReshaped

# <codecell>

# spot checking
print dfCS[(dfCS['Country Name']=='Zambia') & (dfCS['Indicator Code']==indic)].ix[:, '1990':]
print dfReshaped[indic]['Zambia']

# <codecell>

dfReshaped.rename(columns={
'IC.BUS.DISC.XQ':'business_disclosure_index',
'IC.FRM.CMPU.ZS':'firms_competing_against_informal_firms_perc',
'IC.FRM.CORR.ZS':'payments_to_public_officials_perc',
'IC.FRM.INFM.ZS':'do_not_report_all_sales_perc',
'IC.LGL.CRED.XQ':'legal_rights_index',
'IC.LGL.DURS':'time_to_enforce_contract',
'IC.TAX.GIFT.ZS':'bribes_to_tax_officials_perc',
'IQ.CPA.PROP.XQ':'property_rights_rule_governance_rating',
'IQ.CPA.TRAN.XQ':'transparency_accountability_corruption_rating',
'NY.GDP.PCAP.CD':'gdp_per_capita',
'SE.PRM.PRSL.ZS':'primary_school_graduation_perc',
'SI.POV.GINI':'gini_index',
'SL.UEM.TOTL.NE.ZS':'unemployment_perc'
}, inplace=True)

# <codecell>

dfReshaped.to_csv('../data_clean/country_specific_indicators.csv',quoting=csv.QUOTE_ALL)



