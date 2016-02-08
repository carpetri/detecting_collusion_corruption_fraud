countries <- unique(read.csv('../data_clean/clean_country_names_2.csv'))
countries$X <- NULL
names(countries) <-  c('country_clean','country')
inv <- left_join(investigations,countries)
inv$country <- inv$country_clean
inv$country_clean <- NULL
investigations <- inv


