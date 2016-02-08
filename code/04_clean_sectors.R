sectors <- unique(read.csv('../data_clean/sectors.csv'))
sectors$X <- NULL
names(sectors) <- c('sector','major_sector_clean')
investigations <- left_join(investigations,sectors)
investigations$major_sector <- investigations$major_sector_clean
investigations$major_sector_clean <- NULL
investigations$sector <- NULL
summary(investigations)

names(investigations)[which(names(investigations)=='sanctions.case..')] <- 'sanctions.case.number'
names(investigations)[which(names(investigations)=='settlement...resulting.sanction')] <- 'settlements.resulting.sanction'
names(investigations)[which(names(investigations)=='accusation.s.')] <- 'accusations'

names(investigations)[ grep(names(investigations),pattern = '\\.\\.') ]
#write.csv(sort(names(investigations)),row.names=F)
names(investigations)[which(names(investigations)=="draft.fir...evidence.to.slu" )] <- "draft.fir.and.evidence.to.slu"
names(investigations)[which(names(investigations)=="days.open...as.complaint" )] <- "days.open.as.complaint"
names(investigations)[which(names(investigations)=="days.open...as.case" )] <- "days.open.as.case"
names(investigations)[which(names(investigations)=="days.open...total" )] <- "days.open.total"
names(investigations)[which(names(investigations)=="date.sae.returned.to.slu..1st." )] <- "date.sae.returned.to.slu.1st"
names(investigations)[which(names(investigations)=="date.sae.re.submitted.to.oes..1st." )] <- "date.sae.re.submitted.to.oes.1st"
names(investigations)[which(names(investigations)=="date.sae.returned.to.slu..2nd." )] <- "date.sae.returned.to.slu.2nd"
names(investigations)[which(names(investigations)=="date.sae.re.submitted.to.oes..2nd." )] <- "date.sae.re.submitted.to.oes.2nd"

investigations <- investigations[,sort(names(investigations))]
# write.csv(investigations,file='../Data/Investigations/investigations.csv', row.names=F)
write.csv(investigations,file='../data_clean/investigations.csv', row.names=F)
saveRDS(investigations,file='../data_clean/investigations.rds')


