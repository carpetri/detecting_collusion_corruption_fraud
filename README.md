# Detecting Collusion, Corruption, and Fraud

Carlos Petricioli Masters Thesis.

## Summary

The World Bank Group lends billions of dollars each year to fund development projects in its efforts to reduce global poverty. This project helps investigators at the Bank search for patterns of collusion, corruption, and fraud in its contracts data, using models of contract-specific risk. Developing an automated approach to detecting these offenses can help the World Bank efficiently target future investigations.


## The Problem
Contractors providing goods and services on World Bank projects are typically hired through a competitive bidding process. Occasionally, prospective contractors influence the competitive system by colluding with other contractors, bribing government officials, or otherwise manipulating the bidding process. These offenses have far-reaching effects on the price and quality of contract delivery. The World Bank is committed to detecting instances of collusion, corruption, and fraud in order to maximize its global impact.

## Data
To develop contract-level risk models, I incorporated data from multiple sources:

	- Historical data on over 300,000 major contracts funded by World Bank loans from the past 20 years, including such features as company name, country, sector, and total award amount.

	- Annual economic development indicators, collected by the World Bank, for countries and industries within them.

	- World Bank investigations data, covering companies and projects investigated for collusion, corruption or fraud in the past. Includes specific allegations and case outcomes.

## Building Tools for Proactive Investigation

To approach the problem,  an interactive dashboard was developed for World Bank investigators to track a company’s activity across countries, sectors, and time. Using this tool, investigators can:
- Track contract awards companies have received, including under different names (e.g. ACME, Inc. vs. ACME Co.)
- View a risk score for each World Bank contract, as calculated by our contract risk model
- Visualize the immediate neighborhood of the company in its co-award network

## Company Name Disambiguation
Company names are represented by text strings in the data, and a single company may be represented in several very different ways (e.g. ACME Inc. vs. A.C.M.E. Co.)
Company names were reconciled by querying each name on Google and comparing their top ten URL results. Names that had at least 7 links in common were considered to be a single company.

## Co-award networks
A company that works with companies or on projects that have been investigated by the World Bank are more likely to be investigated themselves
Companies that are on many projects together may know each other better and have the opportunity to collude

## Evaluating Contract Risk
This project generates features tracking companies’ historical involvement on World Bank projects within specific countries and sectors, as well as co-award network features for each company. We trained a binary classifier separating past contracts that were investigated by the World Bank from those that were not investigated. We evaluated and compared models using precision, recall, and area under ROC curve. A random forest provided the best results across all metrics on a held-out test set.

## Conclusions and Future Work
- Current data collected by World Bank on contracts is sufficient to forecast risk on future contracts.
- These risk forecasts can allow World Bank investigators to be more proactive in determining which companies, projects and contracts to examine.
- Future analysis will identify separate risk levels for fraud, collusion and corruption




