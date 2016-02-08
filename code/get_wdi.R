library(WDI); library(plyr); library(dplyr); library(tidyr)

countries <- read.csv('../data_clean/clean_country_names.csv') %>% 
  dplyr::rename(country=dirty, country_clean=clean)

#List of the indicators for the project.
indicators <- c('IC.BUS.DISC.XQ', 'IC.FRM.CMPU.ZS', 'IC.FRM.CORR.ZS', 'IC.FRM.INFM.ZS', 'IC.LGL.CRED.XQ','IC.LGL.DURS', 'IC.TAX.GIFT.ZS', 'IQ.CPA.PROP.XQ', 'IQ.CPA.TRAN.XQ', 'NY.GDP.PCAP.CD', 'SE.PRM.PRSL.ZS', 'SI.POV.GINI', 'SL.UEM.TOTL.NE.ZS')

indicators_names <- c('business_disclosure_index', 'firms_competing_against_informal_firms_perc', 'payments_to_public_officials_perc', 'do_not_report_all_sales_perc', 'legal_rights_index', 'time_to_enforce_contract', 'bribes_to_tax_officials_perc', 'property_rights_rule_governance_rating', 'transparency_accountability_corruption_rating', 'gdp_per_capita', 'primary_school_graduation_perc', 'gini_index', 'unemployment_perc')

df <- WDIcache()

df_indicators <- ldply(indicators, function(x){ 
  data.frame(t(
    WDIsearch(string=x,field=c('indicator','name'), cache = df)
    ))})

df_indicators$indicator_name <-  indicators_names

df_wdi <- WDI(country = 'all', indicator = df_indicators$indicator, start = 1990, end=2014,extra=T, cache=df ) %>%
          gather(indicator, value, IC.BUS.DISC.XQ:SL.UEM.TOTL.NE.ZS)  %>% 
          filter(!is.na(value)) %>% 
          filter(region!='Aggregates')

df_wdi <- left_join(df_wdi, select(df_indicators,indicator,indicator_name))

df_wdi_countries <- df_wdi %>% 
  filter(country %in% countries$country) %>% 
  left_join(countries)

write.csv(df_wdi, file = '../data_clean/WDI.csv',row.names = F,quote =T)