library(dplyr);library(ggplot2);library(tidyr);
library(readr); library(plyr);library(stringr);
library(ggthemes);library(extrafont)
loadfonts()
fonts()
rm(list=ls()); gc();gc();gc();gc();

wdi <- read.csv('../data_clean/WDI.csv') %>% tbl_df()

indicators_mean <- wdi %>% group_by(indicator_name, country, region) %>%
  dplyr::summarise(mean_indicator=mean(value, na.rm = T)) 

col_red <- '#a32a2a'
d_ply(indicators_mean,.variables = 'indicator_name',function(df){
  df <- arrange(df,desc(mean_indicator))
  
  df$country <- factor(as.character(df$country), levels = df$country )
  
  tit <-  str_to_title(gsub(pattern = '_',replacement = ' ',x = unique(df$indicator_name))) 
  tit <- gsub(' Perc',' (%)',tit)
  p <- ggplot(df, aes(x=country, y=mean_indicator) ) + 
    geom_bar(stat = 'identity', colour='#585656', size=.1, fill=col_red)+
    facet_wrap(~region, scales = 'free_x', ncol = 2) +
    labs(title=tit, x='Country', y='Mean of the indicator')+
    # theme_bw(base_family = 'Garamond')+
    scale_color_brewer(palette = 'RdGy')+
    theme_hc(base_family = 'Adobe Garamond',base_size = 11) +
    theme(legend.position='bottom', 
          axis.text.x=element_text(angle=90, hjust = 1,vjust = 0.2, size = 6,colour = 'black'))
  # print(p)
  ggsave(p,filename = paste0('../img/wdi_',
                             str_to_lower(unique(df$indicator_name)),
                             '.pdf') , width =7,height = 9.5
         )
})
