if(!grepl('code',getwd()) ) setwd('code/')
library(dplyr);library(ggplot2);library(tidyr);
library(readr); library(plyr);library(stringr);
library(ggthemes);library(extrafont); library(xtable)
library(plotly)
loadfonts()
rm(list=ls()); gc();gc();gc();gc();

wb <- readRDS('../data_clean/wb.rds')

str(wb)

dic <- sapply(wb[,2:48],simplify = F, function(df){
  if(class(df)!='numeric' | class(df)!='logical'){
  x <- df %>% unique() %>% head(15)
  paste0(x, collapse = ', ') 
  }else{
      class(df)
    }
}) %>%  as.data.frame( stringsAsFactors = F) %>% 
  gather(Variable,Values, agreement_type:year)

# write.csv(dic,'../data_clean/major_dict.csv',row.names = F)
dic <- read.csv('../data_clean/major_dict_table.csv')

options(xtable.comment = FALSE)
# print( xtable( dic, label = 'tab_major_dict', caption = 'Major and Historic Awards Dictionary' ) ,
#           include.rownames = F, include.colnames = T,table.placement = 'H',
#          # latex.environments = getOption("xtable.latex.environments", c("")),
#          caption.placement = 'top', file = '../tables/major_historic_dictionary.txt')

wb$fiscal_year %>% unique( ) %>% sort

x <- wb  %>% group_by( region, fiscal_year)  %>%  dplyr::summarise(n=n()) 

library(scales)
p <- ggplot(x,aes(x=fiscal_year,y=n,colour=region, shape=region)) + 
  geom_line(size=.8) + scale_y_continuous(labels = comma)+
  geom_point(size=1.5) + 
  labs(colour='Region', x='Year', y='Number of contracts awarded', shape='Region' )+
  scale_color_brewer(palette = 'Set1')+
  theme_hc(base_family = 'Adobe Garamond')
p
plot_ly()
# plotly_POST(ggplotly(p), filename='../img/major_historic_region.html')
ggsave(filename = '../img/major_historic_region.pdf', plot = p,width = 6,height = 2.8)


x <- wb  %>% 
  group_by( region, fiscal_year)  %>% 
  dplyr::summarise(awarded_amount=sum( award_amount_usd, na.rm=T))

wb %>% filter( fiscal_year=='2013') %>% 
  dplyr::summarise(awarded_amount=mean( award_amount_usd, na.rm=T))
  


p_awarded <- ggplot(x,aes(x=fiscal_year,y=awarded_amount,colour=region, shape=region)) + 
  geom_line(size=.8) + scale_y_continuous(labels = comma)+
  geom_point(size=1.5) + 
  labs(colour='Region', x='Year', y='Total awarded amount\n$USD', shape='Region' )+
  scale_color_brewer(palette = 'Set1')+
  theme_hc(base_family = 'Adobe Garamond')
p_awarded
ggplotly(p_awarded)


ggsave(filename = '../img/major_historic_region_awarded_usd.pdf', plot = p_awarded,width = 6,height = 2.8)

head(wb)

x <- wb  %>% group_by( region, fiscal_year)  %>%  dplyr::summarise(n=n()) 

top_region_awards <- wb %>% 
  left_join(x) %>% 
  group_by(fiscal_year,buyer_country ) %>% 
  dplyr::summarise(n_contracts=n()) %>% 
  ungroup() %>% 
  # group_by(fiscal_year) %>% 
  arrange(fiscal_year,desc(n_contracts)) %>% 
  ddply(.variables = 'fiscal_year', function(df){
      r <- head(df,6)
      r$ranking <- factor(1:6,labels = paste('Top',1:6) )[1:nrow(r)]
      r
    })

library(ggrepel)
library(lubridate)
floor_date( as.Date(2010,format='%Y'),unit = 'year') 

top_region_awards %>% 
  mutate(fiscal_year=floor_date( as.Date(fiscal_year %>% as.character,format='%Y'),unit = 'year') ) %>% 
  ggplot(aes(x= fiscal_year ,
             y=n_contracts)) + 
  # geom_line(size=.8) + 
  # geom_bar(position="dodge")+
  geom_point(colour='red')+
  geom_line(colour='red')+
  geom_text_repel(aes(label=buyer_country), size=2.5)+
  facet_wrap(~ranking, scales = 'free', ncol = 2)+
  scale_y_continuous(labels = comma)+
  scale_x_date(labels = date_format("%Y"),
               date_breaks = '4 years', 
               date_minor_breaks = '1 year', 
               limits = c(as.Date('1988',format='%Y'),as.Date('2014',format='%Y'))
                ) +
  labs( x='Year', y='Number of contracts awarded' )+
  scale_color_brewer(palette = 'Set1')+
  theme_hc(base_family = 'Adobe Garamond', base_size =13)
ggsave(filename = '../img/major_historic_top_contracts_awarded.pdf',
       width = 8,height = 9)

####### TOP MONEY

top_usd <- wb %>% 
  left_join(x) %>% 
  group_by(fiscal_year,buyer_country ) %>% 
  dplyr::summarise(n_awarded_usd=sum(award_amount_usd)) %>% 
  ungroup() %>% 
  # group_by(fiscal_year) %>% 
  arrange(fiscal_year,desc(n_awarded_usd)) %>% 
  ddply(.variables = 'fiscal_year', function(df){
    r <- head(df,6)
    r$ranking <- factor(1:6,labels = paste('Top',1:6) )[1:nrow(r)]
    r
  })

top_usd %>% 
  mutate(fiscal_year=floor_date( as.Date(fiscal_year %>% as.character,format='%Y'),unit = 'year') ) %>% 
  ggplot(aes(x= fiscal_year ,
             y=n_awarded_usd)) + 
  # geom_line(size=.8) + 
  # geom_bar(position="dodge")+
  geom_point(colour='red')+
  geom_line(colour='red')+
  geom_text_repel(aes(label=buyer_country), size=2.1)+
  facet_wrap(~ranking, scales = 'free_x', ncol = 2)+
  scale_y_continuous(labels = comma)+
  scale_x_date(labels = date_format("%Y"),
               date_breaks = '4 years', 
               date_minor_breaks = '1 year', 
               limits = c(as.Date('1988',format='%Y'),as.Date('2014',format='%Y'))
  ) +
  labs( x='Year', y='Total awarded amount\n$USD' )+
  scale_color_brewer(palette = 'Set1')+
  theme_hc(base_family = 'Adobe Garamond', base_size =13)
ggsave(filename = '../img/major_historic_top_usd_awarded.pdf',width = 8,height = 8)


wb %>% filter(competitive==0) %>% select(bid_type) %>% unique

wb %>% filter(competitive==1) %>% select(bid_type) %>% unique

all_entities <- read.csv('../data_clean/all_entities.csv') 

head(all_entities)

uni_len <- function(x){ length(unique(x)) }
all_entities %>% summarise_each(funs(uni_len))

