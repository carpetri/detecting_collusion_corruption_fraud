library(dplyr);library(ggplot2);library(tidyr);
library(readr); library(plyr);library(stringr);
library(ggthemes);library(extrafont)
loadfonts()
fonts()
rm(list=ls()); gc();gc();gc();gc();

investigations <- readRDS('../data_clean/investigations.rds')
head(investigations)
