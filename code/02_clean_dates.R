
investigations <- as.data.frame(investigations)
investigations$date.complaint.received <- as.Date(investigations$date.complaint.received, format='%m/%d/%Y')
investigations$date.complaint.opened <- as.Date(investigations$date.complaint.opened, format='%m/%d/%Y')
investigations$date.preliminary.investigation.opened <- as.Date(investigations$date.preliminary.investigation.opened, format='%m/%d/%Y')
investigations$date.closed <- as.Date(investigations$date.closed, format='%m/%d/%Y')
investigations$case.opened <- as.Date(investigations$case.opened, format='%m/%d/%Y')
investigations$complaint.closed.date <- as.Date(investigations$complaint.closed.date, format='%m/%d/%Y')
investigations$date.closed.fir.sent.to.region  <-  as.Date(investigations$date.closed.fir.sent.to.region , format='%m/%d/%Y')
investigations$date.opened.as.sanctions.case  <- as.Date(investigations$date.opened.as.sanctions.case , format='%m/%d/%Y')
investigations$date.rts.filed.with.oes  <- as.Date( investigations$date.rts.filed.with.oes , format='%m/%d/%Y')
investigations$date.rts.returned.to.slu  <- as.Date(investigations$date.rts.returned.to.slu, format='%m/%d/%Y')
investigations$date.rts.re.submitted  <- as.Date(investigations$date.rts.re.submitted,  format='%m/%d/%Y')
investigations$date.nts.issued <- as.Date(investigations$date.nts.issued,  format='%m/%d/%Y')
investigations$date.settlement.negotiations.initiated <- as.Date(investigations$date.settlement.negotiations.initiated, format='%m/%d/%Y')
investigations$date.settlement.signed.by.respondent  <- as.Date(investigations$date.settlement.signed.by.respondent, format='%m/%d/%Y')
investigations$date.settlement.finalized.by.int.and.sent.to.leg  <- as.Date(investigations$date.settlement.finalized.by.int.and.sent.to.leg,  format='%m/%d/%Y')
investigations$date.settlement.submitted.to.oes <- as.Date(investigations$date.settlement.submitted.to.oes,   format='%m/%d/%Y')
investigations$date.settlement.approved.by.oes <- as.Date(investigations$date.settlement.approved.by.oes,    format='%m/%d/%Y')
investigations$date.sae.returned.to.slu..1st. <- as.Date(investigations$date.sae.returned.to.slu..1st.,     format='%m/%d/%Y')
investigations$date.sae.re.submitted.to.oes..1st. <- as.Date(investigations$date.sae.re.submitted.to.oes..1st., format='%m/%d/%Y')
investigations$date.sae.returned.to.slu..2nd. <- as.Date(investigations$date.sae.returned.to.slu..2nd., format='%m/%d/%Y')
investigations$date.sae.re.submitted.to.oes..2nd.  <- as.Date(investigations$date.sae.re.submitted.to.oes..2nd.,format='%m/%d/%Y') 
investigations$date.of.oes.determination <- as.Date(investigations$date.of.oes.determination, format='%m/%d/%Y') 
investigations$date.of.respondents.response <- as.Date(investigations$date.of.respondents.response, format='%m/%d/%Y') 
investigations$date.of.int.s.reply <- as.Date(investigations$date.of.int.s.reply, format='%m/%d/%Y') 
investigations$date.of.int.s.additional.submission <- as.Date(investigations$date.of.int.s.additional.submission,  format='%m/%d/%Y')
investigations$date.of.sb.hearing <- as.Date(investigations$date.of.sb.hearing, format='%m/%d/%Y') 
investigations$date.of.sb.decision <- as.Date(investigations$date.of.sb.decision, format='%m/%d/%Y')
investigations$date.of.press.release  <- as.Date(investigations$date.of.press.release, format='%m/%d/%Y')
investigations$draft.fir...evidence.to.slu  <- as.Date(investigations$draft.fir...evidence.to.slu, format='%m/%d/%Y')

investigations$cause.letter.sent.date <- NULL
investigations$cause.letter.response.date <- NULL
investigations$date.fir.to.president <- NULL
investigations$date.of.subject...respondent.s.voluntary.restraint <- NULL
investigations$referral.date <- NULL
investigations$redacted.report.date <- NULL
investigations$date.request.for.confidentiality <- NULL
investigations$date.of.appeal.to.sb <- NULL
investigations$date.oes.removes.uncontested.determination <- NULL


unique(investigations$date.oes.removes.uncontested.determination)

head(investigations[,grep(names(investigations), pattern = 'date')])





investigations$fy.complaint.opened <- as.numeric( paste0('20',investigations$fy.complaint.opened))

investigations$fy.case.opened[investigations$fy.case.opened<'90' &investigations$fy.case.opened!='']  <- 
  paste0('20',investigations$fy.case.opened[investigations$fy.case.opened<'90' &investigations$fy.case.opened!=''])
unique(investigations$fy.case.opened)
  
investigations$fy.case.opened[investigations$fy.case.opened>='90' &investigations$fy.case.opened!='']  <- 
  paste0('19',investigations$fy.case.opened[investigations$fy.case.opened>='90' &investigations$fy.case.opened!=''])
unique(investigations$fy.case.opened)

investigations$fy.case.opened <- as.numeric(investigations$fy.case.opened)


#fy.case.closed

investigations$fy.case.closed[investigations$fy.case.closed<'90' &investigations$fy.case.closed!='']  <- 
  paste0('20',investigations$fy.case.closed[investigations$fy.case.closed<'90' &investigations$fy.case.closed!=''])
unique(investigations$fy.case.closed)

investigations$fy.case.closed[investigations$fy.case.closed>='90' &investigations$fy.case.closed!='']  <- 
  paste0('19',investigations$fy.case.closed[investigations$fy.case.closed>='90' &investigations$fy.case.closed!=''])
unique(investigations$fy.case.closed)

investigations$fy.case.closed <- as.numeric(investigations$fy.case.closed)



#fy.opened.as.sanctions.case

investigations$fy.opened.as.sanctions.case[investigations$fy.opened.as.sanctions.case<'90' &investigations$fy.opened.as.sanctions.case!='']  <- 
  paste0('20',investigations$fy.opened.as.sanctions.case[investigations$fy.opened.as.sanctions.case<'90' &investigations$fy.opened.as.sanctions.case!=''])
unique(investigations$fy.opened.as.sanctions.case)

investigations$fy.opened.as.sanctions.case[investigations$fy.opened.as.sanctions.case>='90' &investigations$fy.opened.as.sanctions.case!='']  <- 
  paste0('19',investigations$fy.opened.as.sanctions.case[investigations$fy.opened.as.sanctions.case>='90' &investigations$fy.opened.as.sanctions.case!=''])
unique(investigations$fy.opened.as.sanctions.case)

investigations$fy.opened.as.sanctions.case <- as.numeric(investigations$fy.opened.as.sanctions.case)


head(investigations[,grep(names(investigations), pattern = 'fy')])

summary(investigations)


names(investigations)[which(names(investigations)=='case.opened')] <- 'date.case.opened'
names(investigations)[which(names(investigations)=='date.closed')] <- 'date.case.closed'
names(investigations)[which(names(investigations)=='complaint.closed.date')] <- 'date.complaint.closed'

summary(investigations)

