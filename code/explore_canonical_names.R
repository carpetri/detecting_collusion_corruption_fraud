if(!grepl('code',getwd()) ) setwd('code/')
library(dplyr);library(ggplot2);library(tidyr);
library(readr); library(plyr);library(stringr);
library(ggthemes);library(extrafont); library(xtable)
library(plotly)
loadfonts()
rm(list=ls()); gc();gc();gc();gc();

entities <- read.csv('../Data/Entities/all_entities.csv') %>% dplyr::select(-X)

entities$Name %>% unique() %>%  length()

canonical <- readRDS('../data_clean/wb.rds') %>%  dplyr::select(supplier,canonical_name)

diference <- unique(canonical$supplier) %>% length() - unique(canonical$canonical_name) %>% length() 
100* diference / unique(canonical$supplier) %>% length()

head(canonical,10)

price <- canonical[grepl('PRICE W',canonical$supplier),] %>% unique 
names(price) <- c('Supplier','Canonical Name')
genera_tabla_larga(price,
                   titulo_tabla = 'Entity names disambiguation example',
                   label_tabla ='tab_disambiguation',
                   archivo_salida = '../tables/ex_disambiguation.tex' )

