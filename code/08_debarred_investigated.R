canonical <- read.csv('../Data/Entities/entity_name_matching_combined.csv')
canonical$X <- NULL
head(canonical)
names(canonical) <- c('supplier','canonical_name')
wb <- readRDS('../data_clean/wb.rds')
inv <- readRDS('../data_clean/investigations.rds')
wb$investigated <- wb$canonical_name %in% inv$canonical_name

debarred <- read.csv('../Data/Debarment/world_bank_debarred_companies.csv')
debarred$X <- NULL
head(debarred)

debarred <- left_join(debarred,canonical)
debarred[is.na(debarred$canonical_name),'canonical_name'] <- debarred[is.na(debarred$canonical_name),'supplier']

wb$debarred <- wb$canonical_name %in% debarred$canonical_name

saveRDS(wb,'../data_clean/wb.rds')
