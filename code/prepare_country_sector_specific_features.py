# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Prepare Country/Sector Specific Features For Integration With Procurements
# ===
# 
# Reconciling country names
# --
# * Using the csv file 
# >clean_country_names.csv
# 
#     which resides in worldbank-protected/Data
# 
# 
# 
# 
# 
# 

# <markdowncell>

# Years to keep
# --
# - which years of the country indicators to keep in the final csv?
# - that's defined by the list yearsToKeep = range(19xx, 20xx)
#     

# <codecell>

yearsToKeep = range(1990,2014)

# <markdowncell>

Going to concentrate on only a few country indicators
--
1. IC.BUS.DISC.XQ	Private Sector & Trade: Business environment	
Business extent of disclosure index (0=less disclosure to 10=more disclosure)	Disclosure index measures the extent to which investors are protected through disclosure of ownership and financial information. The index ranges from 0 to 10, with higher values indicating more disclosure.			

2. IC.FRM.CMPU.ZS	Private Sector & Trade: Business environment
Firms competing against unregistered firms (% of firms)

3. IC.FRM.CORR.ZS	Private Sector & Trade: Business environment	Informal payments to public officials (% of firms)

4. IC.FRM.INFM.ZS	Private Sector & Trade: Business environment	Firms that do not report all sales for tax purposes (% of firms)

5. IC.LGL.CRED.XQ	Private Sector & Trade: Business environment	Strength of legal rights index (0=weak to 10=strong)

6. IC.LGL.DURS	Private Sector & Trade: Business environment	Time required to enforce a contract (days)

7. IC.TAX.GIFT.ZS	Private Sector & Trade: Business environment	Firms expected to give gifts in meetings with tax officials (% of firms)

8. IQ.CPA.PROP.XQ	Public Sector: Policy & institutions	CPIA property rights and rule-based governance rating (1=low to 6=high)

9. IQ.CPA.TRAN.XQ	Public Sector: Policy & institutions	CPIA transparency, accountability, and corruption in the public sector rating (1=low to 6=high)

10. NY.GDP.PCAP.CD	Economic Policy & Debt: National accounts: USD at current prices: Aggregate indicators	GDP per capita (current US$)

11. SE.PRM.PRSL.ZS	Education: Efficiency	Persistence to last grade of primary, total (% of cohort)

12. SI.POV.GINI	Poverty: Income distribution	GINI index

13. SL.UEM.TOTL.NE.ZS	Social Protection & Labor: Unemployment	Unemployment, total (% of total labor force) (national estimate)

# <codecell>

import pandas as pd
from numpy import nan

# <codecell>

dfProc = pd.read_csv('../data_clean/Historic_and_Major_awards.csv')
dfCS = pd.read_csv('../data_clean/WDI.csv')

# <codecell>

dfProc.columns

# <markdowncell>

# #Clean country names
# 
# 
# - The file 'clean_country_names.csv' has cleaned names from procuremenets and investigations data
# - I'm reconciling it with the country specific indicators data

# <codecell>

mapOfCountryNames = pd.read_csv('../data_clean/clean_country_names.csv')
# mapOfCountryNames.drop('Unnamed: 0', axis=1, inplace=True)
mapOfCountryNames.to_string().split('\n')

# <codecell>

additionalDirtyNames =  set(dfCS['Country Name'].unique())- set(mapOfCountryNames.dirty)
additionalDirtyNames

# <codecell>

mapOfCountryNames = mapOfCountryNames.append(pd.DataFrame({'dirty':list(additionalDirtyNames), 'clean':nan}), ignore_index=True)
mapOfCountryNames = mapOfCountryNames.drop_duplicates() # is this necessary?.. i may've appended twice before by accident
mapOfCountryNames.sort(columns='dirty').to_string().split('\n')

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

mapOfCountryNames.clean.fillna(mapOfCountryNames.dirty, inplace=True)
mapOfCountryNames.clean.replace(dictForCountryRenaming, inplace=True)
mapOfCountryNames = mapOfCountryNames.drop_duplicates()
mapOfCountryNames
mapOfCountryNames.to_csv('../data_clean/clean_country_names_2.csv',quoting=csv.QUOTE_ALL)

# <markdowncell>

# #Reshape country indicators for integration with other data

# <codecell>

# first, use the previous map of country names to rename countries in the dfCS data
dfCS['Country Name'].replace(mapOfCountryNames.dirty.tolist(), mapOfCountryNames.clean.tolist(), inplace=True)
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


