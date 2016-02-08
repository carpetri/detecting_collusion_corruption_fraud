library(dplyr)

wb <- read.csv('../Data/Procurements/Historic_and_Major_awards.csv',stringsAsFactors =F ) 

wb$canonical_name <- iconv(wb$canonical_name, from='latin1',to= "UTF-8")
head(wb$canonical_name)
wb$canonical_name <- iconv(wb$canonical_name, from='ISO_8859-2',to= "UTF-8")
head(wb$canonical_name)
table(Encoding(wb$canonical_name))
wb$canonical_name <- iconv(wb$canonical_name, from='ASCII',to= "UTF-8")
head(wb$canonical_name)

## some more cleaning for canonical names
dat <- wb

dat[grep(dat$canonical_name,pattern='PRICEWATERHOUSECOOPERS'),'canonical_name']  <- 'PRICE WATER HOUSE COOPERS'
dat[grep(dat$canonical_name,pattern='PRICE W'),'canonical_name']  <- 'PRICE WATER HOUSE COOPERS'
dat[grep(dat$canonical_name,pattern='PRICEW'),'canonical_name']  <- 'PRICE WATER HOUSE COOPERS'


# dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
n.contract[grep(n.contract$canonical_name, pattern = 'AGETIP'), ]

dat[grep(dat$canonical_name,pattern='AGETIP'),'canonical_name']  <- 'AGETIP'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
n.contract[grep(n.contract$canonical_name, pattern = 'UNDP - UNITED NATION DEVELOPMENT PROGRAM|UNDP-IAPSO|CONSULTANTS UNITED NATIONS - UNDP|UNDP-DTCP'),] 
dat[grep(dat$canonical_name,pattern='UNDP - UNITED NATION DEVELOPMENT PROGRAM|UNDP-IAPSO|CONSULTANTS UNITED NATIONS - UNDP|UNDP-DTCP'),'canonical_name']  <- 'UNITED NATION DEVELOPMENT PROGRAM'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
n.contract
n.contract[grep(n.contract$canonical_name, pattern = 'ERNST &'),] 
dat[grep(dat$canonical_name,pattern='ERNST &'),'canonical_name'] <- 'ERNST & YOUNG'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,20)
n.contract[grep(n.contract$canonical_name, pattern = 'IAPSO'),] 
dat[grep(dat$canonical_name,pattern='IAPSO'),'canonical_name'] <- 'IAPSO INTER AGENCY PROCUREMENT SERVICES OFFICE'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,20)

n.contract[grep(n.contract$canonical_name, pattern = 'UNICEF'),] 
dat[grep(dat$canonical_name,pattern='UNICEF'),'canonical_name'] <- 'UNICEF'



n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,20)

write.csv(sort(unique(filter(dat, is.na(canonical_name))$supplier)),file='canonical_nas.csv')

n.contract[setdiff(setdiff(grep(n.contract$canonical_name, pattern = 'SMEC'), 
                   grep(n.contract$canonical_name, pattern = 'CESMEC LTDA')),
                   grep(n.contract$canonical_name, pattern = 'TESMEC S.P.A.') ),] 


dat[setdiff(setdiff(grep(dat$canonical_name, pattern = 'SMEC'), 
                  grep(dat$canonical_name, pattern = 'CESMEC LTDA')),
          grep(dat$canonical_name, pattern = 'TESMEC S.P.A.') )
  ,'canonical_name'] <- 'SMEC INTERNATIONAL PTY'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,20)
n.contract[grep(n.contract$canonical_name,pattern='KPMG'),'canonical_name']
dat[grep(dat$canonical_name,pattern='KPMG'),'canonical_name']  <- 'KPMG'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,11)

dat[grep(dat$canonical_name, pattern = 'COOPERS & LYBRAND'),'canonical_name'] <- 'PRICE WATER HOUSE COOPERS'



n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,11)

n.contract[grep(n.contract$canonical_name, pattern = 'DELOITTE'),] 

dat[grep(dat$canonical_name, pattern = 'DELOITTE'),'canonical_name'] <- 'DELOITTE'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,12)

n.contract[grep(n.contract$canonical_name, pattern = 'UNITED NATION'),] 

dat[grep(dat$canonical_name, pattern = 'UNITED NATIONS'),'canonical_name'] <- 'UNITED NATIONS OFFICE'
dat[grep(dat$canonical_name, pattern = 'UNITED NATION'),'canonical_name'] <- 'UNITED NATIONS'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,12)

n.contract[grep(n.contract$canonical_name, pattern = 'FOOD AND AGRICULTURE'),] 
dat[grep(dat$canonical_name, pattern = 'FOOD AND AGRICULTURE'),'canonical_name']  <- 'FAO - FOOD AND AGRICULTURE ORGANIZATION'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,13)

n.contract[grep(n.contract$canonical_name, pattern = 'BCEOM'),] 
dat[grep(dat$canonical_name, pattern = 'BCEOM'),'canonical_name']  <- 'BCEOM FRENCH ENGINEERING'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,14)
n.contract[grep(n.contract$canonical_name, pattern = 'SOFRECO'),] 
dat[grep(dat$canonical_name, pattern = 'SOFRECO'),'canonical_name']  <- 'SOFRECO'



n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,16)
n.contract[grep(n.contract$canonical_name, pattern = 'TOYOTA'),] 
dat[grep(dat$canonical_name, pattern = 'TOYOTA'),'canonical_name']  <- 'TOYOTA'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,17)
n.contract[grep(n.contract$canonical_name, pattern = 'ARTHUR ANDERSEN'),] 
dat[grep(dat$canonical_name, pattern = 'ARTHUR ANDERSEN'),'canonical_name']  <- 'ARTHUR ANDERSEN LLC'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,18)
n.contract[grep(n.contract$canonical_name, pattern = 'MITSHUBISHI'),] 
dat[grep(dat$canonical_name, pattern = 'MITSHUBISHI'),'canonical_name']  <- 'MITSHUBISHI'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,19)
n.contract[grep(n.contract$canonical_name, pattern = 'ITOCHU'),] 
dat[grep(dat$canonical_name, pattern = 'ITOCHU'),'canonical_name']  <- 'ITOCHU CORPORATION'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,20)
n.contract[grep(n.contract$canonical_name, pattern = 'LOUIS BERGER'),] 
dat[grep(dat$canonical_name, pattern = 'LOUIS BERGER'),'canonical_name']  <- 'LOUIS BERGER INTERNATIONAL'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,22)
n.contract[grep(n.contract$canonical_name, pattern = 'NESTOR'),] 
dat[grep(dat$canonical_name, pattern = 'NESTOR PHARMACEUTICALS'),'canonical_name']  <- 'NESTOR PHARMACEUTICALS'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,22)
n.contract[grep(n.contract$canonical_name, pattern = 'BOOZ ALLEN & HAMILTON|BOOZ ALLEN'),] 
dat[grep(dat$canonical_name, pattern = 'BOOZ ALLEN & HAMILTON|BOOZ ALLEN'),'canonical_name']  <- 'BOOZ ALLEN AND HAMILTON'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,22)
n.contract[grep(n.contract$canonical_name, pattern = 'BOOZ ALLEN & HAMILTON|BOOZ ALLEN'),] 
dat[grep(dat$canonical_name, pattern = 'BOOZ ALLEN & HAMILTON|BOOZ ALLEN'),'canonical_name']  <- 'BOOZ ALLEN AND HAMILTON'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,23)
n.contract[grep(n.contract$canonical_name, pattern = 'CHINA COMMUNICATIONS'),] 
dat[grep(dat$canonical_name, pattern = 'CHINA COMMUNICATIONS'),'canonical_name']  <- 'CHINA COMMUNICATIONS CONSTRUCTIONS CO'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,24)
n.contract[grep(n.contract$canonical_name, pattern = 'STUDI I'),] 
dat[grep(dat$canonical_name, pattern = 'STUDI I'),'canonical_name']  <- 'STUDI INTERNATIONAL'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,25)
n.contract[grep(n.contract$canonical_name, pattern = 'CHINA NATIONAL ELECTRIC'),] 
unique(dat[grep(dat$canonical_name, pattern = 'CHINA NATIONAL ELECTRIC'),'supplier'])
dat[grep(dat$canonical_name, pattern = 'CHINA NATIONAL ELECTRIC'),'canonical_name']  <- 'CHINA NATIONAL ELECTRICAL ENGINEERING'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,25)
n.contract[grep(n.contract$canonical_name, pattern = 'CHINA NATIONAL ELECTRIC'),] 

dat[grep(dat$canonical_name, pattern = 'CHINA NATIONAL ELECTRICAL ENGINEERING'),'canonical_name']  <- 'CHINA NATIONAL ELECTRICAL ENGINEERING'
dat[setdiff( grep(dat$canonical_name, pattern = 'CHINA NATIONAL ELECTRIC'),
  grep(dat$canonical_name, pattern = 'CHINA NATIONAL ELECTRICAL ENGINEERING'))
  ,'canonical_name']  <- 'CHINA NATIONAL ELECTRIC WIRE AND CABLE IMPORT AND EdatPORT CO'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,26)

n.contract[setdiff(grep(n.contract$canonical_name, pattern = 'COLAS'),
                   grep(n.contract$canonical_name, pattern = 'NICOLAS|ECOLAS|MICOLAS|AGRICOLAS'))
           ,] 

dat[setdiff(grep(dat$canonical_name, pattern = 'COLAS'),
          grep(dat$canonical_name, pattern = 'NICOLAS|ECOLAS|MICOLAS|AGRICOLAS')),
  'canonical_name']  <- 'COLAS'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,27)

n.contract[grep(n.contract$canonical_name, pattern = 'SFCE'),] 

dat[grep(dat$canonical_name, pattern = 'SFCE'),'canonical_name']  <- 'SFCE - SOCIETE FRANCAISE DE COMMERCE EUROPEAN'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,28)

n.contract[grep(n.contract$canonical_name, pattern = 'DHV '),] 

dat[grep(dat$canonical_name, pattern = 'DHV '),'canonical_name']  <- 'DHV CONSULTANTS BV'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,28)

n.contract[grep(n.contract$canonical_name, pattern = 'DHV '),] 

dat[grep(dat$canonical_name, pattern = 'DHV '),'canonical_name']  <- 'DHV CONSULTANTS BV'



n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,29)

n.contract[grep(n.contract$canonical_name, pattern = 'IICA'),] 

dat[grep(dat$canonical_name, pattern = 'IICA'),]  <- 'IICA'


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,30)

n.contract[grep(n.contract$canonical_name, pattern = 'BRAC$|\\(BRAC\\)|BRAC-|BRACCO'),]
          
dat[grep(dat$canonical_name, pattern = 'BRAC$|\\(BRAC\\)|BRAC-|BRACCO'),'canonical_name']  <- 'BRAC-BANGLADESH RURAL ADVANCEMENT COMMITTEE'

n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,31)

dat[grep(dat$canonical_name, pattern = 'HUBEI'),]  




dat$canonical_name <- iconv(dat$canonical_name, from='latin1',to= "UTF-8",'')
head(dat$canonical_name)
dat$canonical_name <- iconv(dat$canonical_name, from='ISO_8859-2',to= "UTF-8",'')
head(dat$canonical_name)
table(Encoding(dat$canonical_name))
dat$canonical_name <- iconv(dat$canonical_name, from='ASCII',to= "UTF-8",'')
head(dat$canonical_name)


n.contract <- dat %.% group_by(canonical_name) %.% dplyr::summarise(n=n()) %.% arrange(-n)
head(n.contract,29)


dat$canonical_name[is.na(dat$canonical_name)] <- dat$supplier[is.na(dat$canonical_name)]




del <- paste( collapse = '|',c("^\ ", "^  ","^   ","^ \\",'^\\"','\\',
                               "#"  ,"^\'","^\"" , "^\\(",  "\\*" ,
                               "^\\," , "^\\."  , "?",  "^\\@" ,"^\\`" ) )

x <- gsub(dat$canonical_name, pattern =del, replace='' )
xx <- gsub(x, pattern =del, replace='' )
xx[grep(xx,pattern = '^#')] <- gsub(xx[grep(xx,pattern = '^#')], pattern = '#', replace='')
xx[grep(xx,pattern = '^ ')] <-gsub(xx[grep(xx,pattern = '^ ')], pattern = '^ ' , replace='') 
xx[grep(xx,pattern = '^\\?\\?\\? ')] <- gsub(xx[grep(xx,pattern = '^\\?\\?\\? ')], pattern = '^\\?\\?\\? ' , replace='') 
xx[grep(xx,pattern = '^[0-9]')]  <- paste0('#',xx[grep(xx,pattern = '^[0-9]')]) 

# xx[grep(xx,pattern = '^\U3e35383c')]   <- c('OZTAS INSAAT INSAAT MALZEMELERI TICARET ANONIM SIRKETI',
#                                             'OSB CONSULTING GMBH/AUSTRIA & SEOR COMPANY /NETHERLANDS',
#                                             'OSB CONSULTING GMBH/AUSTRIA & SEOR COMPANY /NETHERLANDS')
# xx[grep(xx,pattern = '^\U3e35653c')]  <- 'ANGELO ROBERTO GOMES SILVA'
# xx[grep(xx,pattern = '^\U3e38653c')]  <- c('EDDY BYNENS','ERB-RINA-KALY TAO')


first <- substr(xx,start = 1, stop = 1)
sort(unique(first))

wb$canonical_name <- dat$canonical_name
wb$first_letter <- first 

wb %.% group_by(first_letter) %.% dplyr::summarise(n=n())

wb <- arrange(wb,canonical_name)





# canonical_names <- sort(unique(wb$canonical_name))
# cache('canonical_names')
# 




# 
# 
# 
# 
# 
# #######other thiongs
# 
# unique(projects$project_id)
# 
# df <- wb%.% filter(project_id %in% 'P001340') %.% select(supplier,project_id)
# df <- unique(df)
# df
# net <- unique(dplyr::left_join(df, df, by = 'project_id')) 
# head(net)
# net.1 <- dplyr::filter(net, supplier.x !=net$supplier.y)
# 
# net.1$project_id <- NULL
# names(net.1) <- c('Source','Target')
# 
# unique(net.1$Source)
# unique(net.1$Target)
# net <- net.1
# 
# cache('net')
# 
# df <- wb %.% filter(buyer_country=='China' & major_sector_clean=='energy/mining' & fiscal_year==2013) %.%
#   group_by(buyer_country,major_sector_clean, fiscal_year ) %.% 
#   #  mutate(award_amount_usd=award_amount_usd, scale = F, na.rm=T))%.%
#   select(award_amount_usd)
# 
# ggplot(df,aes(x=award_amount_usd+1) ) + geom_histogram(  binwidth=.5,alpha=.8,
#                                                          colour="black", fill="grey" )+ 
#   scale_x_log10() + 
#   geom_density(alpha=.2, fill="red") +
#   facet_wrap(~fiscal_year) + 
#   ggtitle(paste('Awarded amount distribution in','China','and sector','energy/mining'))+
#   xlab('Awarded amount ($USD)')+ylab('')
#  
# 
