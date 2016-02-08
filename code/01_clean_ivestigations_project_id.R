library(dplyr);library(ggplot2);library(tidyr);
library(readr); library(plyr);library(stringr);
library(ggthemes);library(extrafont)
loadfonts()
fonts()
rm(list=ls()); gc();gc();gc();gc();

investigations <- unique(tbl_df(read.csv('../Data/Investigations/libre-office-encoded-CMS-Management-Report-20140626-103453.csv')))
#wb <- read.csv('../../worldbank/Data/Procurements/MDB_data/WorldBank/Historic_and_Major_awards.csv')
#cache('wb')

names(investigations) <- tolower(names(investigations))
table(investigations$complaint.status)

print(table(investigations$complaint.status))
head(as.data.frame(investigations))

investigations$fy.complaint.opened <- gsub(investigations$fy.complaint.opened, pattern = 'FY', replacement = '')
investigations$fy.case.opened  <- gsub(investigations$fy.case.opened, pattern = 'FY', replacement = '') 
investigations$fy.case.closed <- gsub(investigations$fy.case.closed, pattern='FY', replacement = '')
investigations$fy.opened.as.sanctions.case  <- gsub(investigations$fy.opened.as.sanctions.case, pattern='FY', replacement = '')

unique(investigations$fy.complaint.opened)
unique(investigations$fy.case.opened)
unique(investigations$fy.case.closed)
unique(investigations$fy.opened.as.sanctions.case)


summary(investigations)

investigations$project.number <- gsub(investigations$project.number, pattern=' ',replacement = '')
investigations$project.number <- gsub(investigations$project.number, pattern=',',replacement = '|')
investigations$project.number <- gsub(investigations$project.number, pattern='and',replacement = '|')
investigations$project.number <- gsub(investigations$project.number, pattern='/',replacement = '|')
investigations$project.number <- gsub(investigations$project.number, pattern=';',replacement = '|')
investigations$project.number <- gsub(investigations$project.number, pattern='respectively',replacement = '')
investigations$project.number  <- gsub(investigations$project.number, pattern='InfrastructureReconstructionFinancingFacility\\(IRFF\\|', replacement='')
investigations$project.number <- gsub(investigations$project.number,pattern = '(PowerRehabilitationProject-P035076)', replacement = 'P035076')
investigations$project.number <- gsub(investigations$project.number, pattern='HealthServicesImprovementProject\\(P074027\\)',replacement = 'P074027')
investigations$project.number <- gsub(investigations$project.number, pattern = "1.P109224\n2.P044695", replacement='P109224|P044695')

investigations$project.number <- gsub(investigations$project.number, pattern='PrivateSectorCompetitivenessIIProject-P083809',replacement = 'P083809')

investigations$project.number <- gsub(investigations$project.number, pattern='EDP2Project\\(P078113\\)',replacement = 'P078113')
investigations$project.number <- gsub(investigations$project.number, pattern='IFC:548425',replacement = 'P548425')

investigations$project.number <- gsub(investigations$project.number, pattern='P087945.',replacement = 'P087945')
investigations$project.number <- gsub(investigations$project.number, pattern='28472|26174',replacement = 'P28472|P26174')
investigations$project.number <- gsub(investigations$project.number, pattern='\\|$',replacement = '')
investigations$project.number <- gsub(investigations$project.number, pattern=')$',replacement = '')
investigations$project.number <- gsub(investigations$project.number, pattern='\\(',replacement = '')
investigations$project.number <- gsub(investigations$project.number, pattern='\n',replacement = '|')








