"""
Enhanced table metadata and available sectors/industries for the financial data analyzer.
This file contains the schema definitions and reference data used by the FinancialDataAnalyzer.
"""

from typing import Dict, Any, List

# Available sectors for filtering
AVAILABLE_SECTORS = sorted([
    "Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate",
    "Technology", "Utilities"
])

# Available industries for filtering
AVAILABLE_INDUSTRIES = sorted([
    "Steel", "Silver", "Other Precious Metals", "Gold", "Copper", "Aluminum",
    "Paper, Lumber & Forest Products", "Industrial Materials", "Construction Materials",
    "Chemicals - Specialty", "Chemicals", "Agricultural Inputs", "Telecommunications Services",
    "Internet Content & Information", "Publishing", "Broadcasting", "Advertising Agencies",
    "Entertainment", "Travel Lodging", "Travel Services", "Specialty Retail", "Luxury Goods",
    "Home Improvement", "Residential Construction", "Department Stores",
    "Personal Products & Services", "Leisure", "Gambling, Resorts & Casinos",
    "Furnishings, Fixtures & Appliances", "Restaurants", "Auto - Parts", "Auto - Manufacturers",
    "Auto - Recreational Vehicles", "Auto - Dealerships", "Apparel - Retail",
    "Apparel - Manufacturers", "Apparel - Footwear & Accessories", "Packaging & Containers",
    "Tobacco", "Grocery Stores", "Discount Stores", "Household & Personal Products",
    "Packaged Foods", "Food Distribution", "Food Confectioners", "Agricultural Farm Products",
    "Education & Training Services", "Beverages - Wineries & Distilleries",
    "Beverages - Non-Alcoholic", "Beverages - Alcoholic", "Uranium", "Solar",
    "Oil & Gas Refining & Marketing", "Oil & Gas Midstream", "Oil & Gas Integrated",
    "Oil & Gas Exploration & Production", "Oil & Gas Equipment & Services",
    "Oil & Gas Energy", "Oil & Gas Drilling", "Coal", "Shell Companies",
    "Investment - Banking & Investment Services", "Insurance - Specialty",
    "Insurance - Reinsurance", "Insurance - Property & Casualty", "Insurance - Life",
    "Insurance - Diversified", "Insurance - Brokers", "Financial - Mortgages",
    "Financial - Diversified", "Financial - Data & Stock Exchanges",
    "Financial - Credit Services", "Financial - Conglomerates", "Financial - Capital Markets",
    "Banks - Regional", "Banks - Diversified", "Banks", "Asset Management",
    "Asset Management - Bonds", "Asset Management - Income", "Asset Management - Leveraged",
    "Asset Management - Cryptocurrency", "Asset Management - Global", "Medical - Specialties",
    "Medical - Pharmaceuticals", "Medical - Instruments & Supplies", "Medical - Healthcare Plans",
    "Medical - Healthcare Information Services", "Medical - Equipment & Services",
    "Medical - Distribution", "Medical - Diagnostics & Research", "Medical - Devices",
    "Medical - Care Facilities", "Drug Manufacturers - Specialty & Generic",
    "Drug Manufacturers - General", "Biotechnology", "Waste Management", "Trucking",
    "Railroads", "Aerospace & Defense", "Marine Shipping", "Integrated Freight & Logistics",
    "Airlines, Airports & Air Services", "General Transportation",
    "Manufacturing - Tools & Accessories", "Manufacturing - Textiles",
    "Manufacturing - Miscellaneous", "Manufacturing - Metal Fabrication",
    "Industrial - Distribution", "Industrial - Specialties",
    "Industrial - Pollution & Treatment Controls", "Environmental Services",
    "Industrial - Machinery", "Industrial - Infrastructure Operations",
    "Industrial - Capital Goods", "Consulting Services", "Business Equipment & Supplies",
    "Staffing & Employment Services", "Rental & Leasing Services",
    "Engineering & Construction", "Security & Protection Services",
    "Specialty Business Services", "Construction", "Conglomerates",
    "Electrical Equipment & Parts", "Agricultural - Machinery",
    "Agricultural - Commodities/Milling", "REIT - Specialty", "REIT - Retail",
    "REIT - Residential", "REIT - Office", "REIT - Mortgage", "REIT - Industrial",
    "REIT - Hotel & Motel", "REIT - Healthcare Facilities", "REIT - Diversified",
    "Real Estate - Services", "Real Estate - Diversified", "Real Estate - Development",
    "Real Estate - General", "Information Technology Services", "Hardware, Equipment & Parts",
    "Computer Hardware", "Electronic Gaming & Multimedia", "Software - Services",
    "Software - Infrastructure", "Software - Application", "Semiconductors",
    "Media & Entertainment", "Communication Equipment", "Technology Distributors",
    "Consumer Electronics", "Renewable Utilities", "Regulated Water", "Regulated Gas",
    "Regulated Electric", "Independent Power Producers", "Diversified Utilities",
    "General Utilities"
])

def get_enhanced_table_metadata() -> Dict[str, Dict[str, Any]]:
    """Enhanced table metadata with ACTUAL column names verified from data"""
    return {
        'company_profiles': {
            'description': 'General company information: symbol, companyName, sector, industry, description, location, website, market cap (mktCap), stock price. Located in financial_data schema.',
            'primary_sort_column': 'mktcap', 
            'sort_direction': 'DESC', 
            'date_columns': [],
            'key_columns': ['beta', 'changes', 'cik', 'companyname', 'country', 'currency','description', 'isin', 'lastdiv', 'mktcap','price', 'range', 'sector','symbol'],
            'filter_hints': 'Filter by sector, industry, or market cap ranges. Always use financial_data.company_profiles.',
            'is_time_series': False,
            'time_column': None, 
            'period_column': None,
            'special_notes': 'Primary reference table for company information. Always join with other tables on symbol. Use financial_data.company_profiles with lowercase column names.',
            'actual_columns': ['beta', 'changes', 'cik', 'companyname', 'country', 'currency','description', 'isin', 'lastdiv', 'mktcap','price', 'range', 'sector','symbol']
        },

        'income_statements': {
            'description': 'Financial income performance: revenue, grossprofit, operatingincome, netincome, eps, epsdiluted, ebitda. Uses calendaryear. Located in financial_data schema.',
            'primary_sort_column': 'revenue', 
            'sort_direction': 'DESC', 
            'date_columns': ['date', 'fillingdate', 'accepteddate'],
            'time_column': 'calendaryear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'revenue', 'grossprofit', 'operatingincome', 'netincome', 'eps', 'epsdiluted', 'ebitda', 'calendaryear', 'period', 'cik'],
            'filter_hints': 'Default to calendaryear = 2024 and period = \'FY\' for annual data. Always use financial_data.income_statements.',
            'is_time_series': True,
            'special_notes': 'Uses calendaryear (NOT fiscalyear like other tables). Use financial_data.income_statements with lowercase column names.',
            'actual_columns': ['symbol', 'date', 'period', 'revenue', 'grossprofit', 'operatingincome', 'netincome', 'accepteddate', 'calendaryear', 'cik', 'costandexpenses', 'costofrevenue', 'depreciationandamortization', 'ebitda', 'ebitdaratio', 'eps', 'epsdiluted', 'generalandadministrativeexpenses', 'grossprofitgratio', 'incomebeforetax', 'incomebeforetaxratio', 'incometaxexpense', 'interestexpense', 'interestincome', 'netincomeratio', 'operatingexpenses', 'operatingincomeratio', 'otherexpenses', 'reportedcurrency', 'researchanddevelopmentexpenses', 'sellingandmarketingexpenses', 'sellinggeneralandadministrativeexpenses', 'totalotherinexcomepensesnet', 'weightedaverageshsout', 'weightedaverageshsoutdil']
        },
        'balance_sheets': {
            'description': 'Company assets, liabilities, and equity: totalAssets, totalLiabilities, totalEquity, cashAndCashEquivalents, totalDebt, etc. Uses calendarYear.',
            'primary_sort_column': 'totalAssets', 
            'sort_direction': 'DESC', 
            'date_columns': ['date', 'fillingDate', 'acceptedDate'],
            'time_column': 'calendarYear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'totalAssets', 'totalLiabilities', 'totalEquity', 'cashAndCashEquivalents', 'totalDebt', 'netDebt', 'calendarYear', 'period', 'cik'],
            'filter_hints': 'Default to calendarYear = 2024 and period = "FY" for annual data.',
            'is_time_series': True,
            'special_notes': 'Uses calendarYear (NOT fiscalYear). Column name is totalEquity, not totalStockholderEquity.',
            'actual_columns': ['symbol', 'date', 'period', 'totalAssets', 'totalLiabilities', 'totalEquity', 'acceptedDate', 'accountPayables', 'accumulatedOtherComprehensiveIncomeLoss', 'calendarYear', 'capitalLeaseObligations', 'cashAndCashEquivalents', 'cashAndShortTermInvestments', 'cik', 'commonStock', 'deferredRevenue', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'goodwill', 'goodwillAndIntangibleAssets', 'intangibleAssets', 'inventory', 'longTermDebt', 'longTermInvestments', 'minorityInterest', 'netDebt', 'netReceivables', 'otherAssets', 'otherCurrentAssets', 'otherCurrentLiabilities', 'otherLiabilities', 'otherNonCurrentAssets', 'otherNonCurrentLiabilities', 'othertotalStockholdersEquity', 'preferredStock', 'propertyPlantEquipmentNet', 'reportedCurrency', 'retainedEarnings', 'shortTermDebt', 'shortTermInvestments', 'taxAssets', 'taxPayables', 'totalCurrentAssets', 'totalCurrentLiabilities', 'totalDebt', 'totalInvestments', 'totalLiabilities']
        },
        'cash_flow_statements': {
            'description': 'Cash inflows and outflows: operatingCashFlow, netCashUsedForInvestingActivites, netCashUsedProvidedByFinancingActivities, freeCashFlow, etc. Uses calendarYear.',
            'primary_sort_column': 'operatingCashFlow', 
            'sort_direction': 'DESC', 
            'date_columns': ['date', 'fillingDate', 'acceptedDate'],
            'time_column': 'calendarYear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'operatingCashFlow', 'netCashUsedForInvestingActivites', 'netCashUsedProvidedByFinancingActivities', 'freeCashFlow', 'netChangeInCash', 'calendarYear', 'period', 'cik'],
            'filter_hints': 'Default to calendarYear = 2024 and period = "FY" for annual data.',
            'is_time_series': True,
            'special_notes': 'Uses calendarYear (NOT fiscalYear). Note long column names: netCashUsedForInvestingActivites, netCashUsedProvidedByFinancingActivities.',
            'actual_columns': ['symbol', 'date', 'period', 'operatingCashFlow', 'acceptedDate', 'accountsPayables', 'accountsReceivables', 'acquisitionsNet', 'calendarYear', 'capitalExpenditure', 'cashAtBeginningOfPeriod', 'cashAtEndOfPeriod', 'changeInWorkingCapital', 'cik', 'commonStockIssued', 'commonStockRepurchased', 'deferredIncomeTax', 'depreciationAndAmortization', 'effectOfForexChangesOnCash', 'freeCashFlow', 'inventory', 'investmentsInPropertyPlantAndEquipment', 'netCashProvidedByOperatingActivities', 'netCashUsedForInvestingActivites', 'netCashUsedProvidedByFinancingActivities', 'netChangeInCash', 'netIncome', 'otherFinancingActivites', 'otherInvestingActivites', 'otherNonCashItems', 'otherWorkingCapital', 'purchasesOfInvestments', 'reportedCurrency', 'salesMaturitiesOfInvestments', 'stockBasedCompensation']
        },

        
        # TTM (Trailing Twelve Months) Financial Statements
        'income_statements_ttm': {
            'description': 'TTM Income statements: revenue, costOfRevenue, grossProfit, operatingIncome, netIncome, eps, epsDiluted, ebitda. Uses fiscalYear and period.',
            'primary_sort_column': 'revenue', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'fiscalYear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'revenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome', 'eps', 'epsDiluted', 'ebitda', 'fiscalYear', 'period', 'cik'],
            'filter_hints': 'For latest revenue, use WHERE fiscalYear = (SELECT MAX(fiscalYear) FROM income_statements_ttm). TTM periods are Q1, Q2, Q3, Q4 (NOT FY). To get most recent data, use ORDER BY fiscalYear DESC, period DESC or ORDER BY date DESC.',
            'is_time_series': True,
            'special_notes': 'TTM data uses fiscalYear (NOT calendarYear). Periods are quarterly (Q1, Q2, Q3, Q4), never "FY".',
            'actual_columns': ['symbol', 'date', 'fiscalYear', 'period', 'revenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome', 'eps', 'epsDiluted', 'ebitda', 'acceptedDate', 'bottomLineNetIncome', 'cik', 'costAndExpenses', 'depreciationAndAmortization', 'ebit', 'generalAndAdministrativeExpenses', 'incomeBeforeTax', 'incomeTaxExpense', 'interestExpense', 'interestIncome', 'netIncomeDeductions', 'netIncomeFromContinuingOperations', 'netIncomeFromDiscontinuedOperations', 'netInterestIncome', 'nonOperatingIncomeExcludingInterest', 'operatingExpenses', 'otherAdjustmentsToNetIncome', 'otherExpenses', 'reportedCurrency', 'researchAndDevelopmentExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'totalOtherIncomeExpensesNet']
        },
        'balance_sheets_ttm': {
            'description': 'TTM Balance sheets: totalAssets, totalLiabilities, totalEquity, cashAndCashEquivalents, totalDebt, netDebt. Uses fiscalYear and period.',
            'primary_sort_column': 'totalAssets', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'fiscalYear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'totalAssets', 'totalLiabilities', 'totalEquity', 'cashAndCashEquivalents', 'totalDebt', 'netDebt', 'fiscalYear', 'period', 'cik'],
            'filter_hints': 'For latest data, use WHERE fiscalYear = (SELECT MAX(fiscalYear) FROM balance_sheets_ttm). TTM periods are Q1, Q2, Q3, Q4 (NOT FY). To get most recent data, use ORDER BY fiscalYear DESC, period DESC or ORDER BY date DESC.',
            'is_time_series': True,
            'special_notes': 'TTM data uses fiscalYear (NOT calendarYear). Periods are quarterly (Q1, Q2, Q3, Q4), never "FY".',
            'actual_columns': ['symbol', 'date', 'fiscalYear', 'period', 'totalAssets', 'totalLiabilities', 'totalEquity', 'cashAndCashEquivalents', 'totalDebt', 'netDebt', 'acceptedDate', 'accountPayables', 'accountsReceivables', 'accruedExpenses', 'accumulatedOtherComprehensiveIncomeLoss', 'additionalPaidInCapital', 'capitalLeaseObligations', 'capitalLeaseObligationsCurrent', 'cashAndShortTermInvestments', 'cik', 'commonStock', 'deferredRevenue', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'goodwill', 'goodwillAndIntangibleAssets', 'intangibleAssets', 'inventory', 'longTermDebt', 'longTermInvestments', 'minorityInterest', 'netReceivables', 'otherAssets', 'otherCurrentAssets', 'otherCurrentLiabilities', 'otherLiabilities', 'otherNonCurrentAssets', 'otherNonCurrentLiabilities', 'otherPayables', 'otherReceivables', 'otherTotalStockholdersEquity', 'preferredStock', 'prepaids', 'propertyPlantEquipmentNet', 'reportedCurrency', 'retainedEarnings', 'shortTermDebt', 'shortTermInvestments', 'taxAssets', 'taxPayables', 'totalCurrentAssets', 'totalCurrentLiabilities', 'totalInvestments', 'totalLiabilities']
        },
        'cash_flow_statements_ttm': {
            'description': 'TTM Cash flow statements: netIncome, operatingCashFlow, freeCashFlow, netCashProvidedByFinancingActivities, etc. Uses fiscalYear and period.',
            'primary_sort_column': 'operatingCashFlow', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'fiscalYear', 
            'period_column': 'period',
            'key_columns': ['symbol', 'netIncome', 'operatingCashFlow', 'freeCashFlow', 'netCashProvidedByFinancingActivities', 'netCashProvidedByInvestingActivities', 'fiscalYear', 'period', 'cik'],
            'filter_hints': 'For latest data, use WHERE fiscalYear = (SELECT MAX(fiscalYear) FROM cash_flow_statements_ttm). TTM periods are Q1, Q2, Q3, Q4 (NOT FY). To get most recent data, use ORDER BY fiscalYear DESC, period DESC or ORDER BY date DESC.',
            'is_time_series': True,
            'special_notes': 'TTM data uses fiscalYear (NOT calendarYear). Periods are quarterly (Q1, Q2, Q3, Q4), never "FY".',
            'actual_columns': ['symbol', 'date', 'fiscalYear', 'period', 'netIncome', 'operatingCashFlow', 'freeCashFlow', 'acceptedDate', 'accountsPayables', 'accountsReceivables', 'acquisitionsNet', 'capitalExpenditure', 'cashAtBeginningOfPeriod', 'cashAtEndOfPeriod', 'changeInWorkingCapital', 'cik', 'commonDividendsPaid', 'commonStockIssuance', 'commonStockRepurchased', 'deferredIncomeTax', 'depreciationAndAmortization', 'effectOfForexChangesOnCash', 'incomeTaxesPaid', 'interestPaid', 'inventory', 'investmentsInPropertyPlantAndEquipment', 'longTermNetDebtIssuance', 'netCashProvidedByFinancingActivities', 'netCashProvidedByInvestingActivities', 'netCashProvidedByOperatingActivities', 'netChangeInCash', 'netCommonStockIssuance', 'netDebtIssuance', 'netDividendsPaid', 'netPreferredStockIssuance', 'netStockIssuance', 'otherFinancingActivities', 'otherInvestingActivities', 'otherNonCashItems', 'otherWorkingCapital', 'preferredDividendsPaid', 'purchasesOfInvestments', 'reportedCurrency', 'salesMaturitiesOfInvestments', 'shortTermNetDebtIssuance', 'stockBasedCompensation']
        },
        'financial_ratios_ttm': {
            'description': 'TTM Financial ratios: currentRatio, quickRatio, grossProfitMargin, netProfitMargin, debtToEquityRatio, etc. Uses date for time series.',
            'primary_sort_column': 'currentRatioTTM', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'date', 
            'period_column': None,
            'key_columns': ['symbol', 'currentRatioTTM', 'quickRatioTTM', 'grossProfitMarginTTM', 'netProfitMarginTTM', 'debtToEquityRatioTTM', 'date'],
            'filter_hints': 'For latest data, use ORDER BY date DESC. TTM ratios are current as of the date field.',
            'is_time_series': True,
            'special_notes': 'TTM ratios are current performance metrics. Uses date field for time series.',
            'actual_columns': ['symbol', 'assetTurnoverTTM', 'bookValuePerShareTTM', 'bottomLineProfitMarginTTM', 'capexPerShareTTM', 'capitalExpenditureCoverageRatioTTM', 'cashPerShareTTM', 'cashRatioTTM', 'continuousOperationsProfitMarginTTM', 'currentRatioTTM', 'debtServiceCoverageRatioTTM', 'debtToAssetsRatioTTM', 'debtToCapitalRatioTTM', 'debtToEquityRatioTTM', 'debtToMarketCapTTM', 'dividendPaidAndCapexCoverageRatioTTM', 'dividendPayoutRatioTTM', 'dividendPerShareTTM', 'dividendYieldTTM', 'ebitMarginTTM', 'ebitdaMarginTTM', 'ebtPerEbitTTM', 'effectiveTaxRateTTM', 'enterpriseValueMultipleTTM', 'enterpriseValueTTM', 'financialLeverageRatioTTM', 'fixedAssetTurnoverTTM', 'forwardPriceToEarningsGrowthRatioTTM', 'freeCashFlowOperatingCashFlowRatioTTM', 'freeCashFlowPerShareTTM', 'grossProfitMarginTTM', 'interestCoverageRatioTTM', 'interestDebtPerShareTTM', 'inventoryTurnoverTTM', 'longTermDebtToCapitalRatioTTM', 'netIncomePerEBTTTM', 'netIncomePerShareTTM', 'netProfitMarginTTM', 'operatingCashFlowCoverageRatioTTM', 'operatingCashFlowPerShareTTM', 'operatingCashFlowRatioTTM', 'operatingCashFlowSalesRatioTTM', 'operatingProfitMarginTTM', 'payablesTurnoverTTM', 'pretaxProfitMarginTTM', 'priceToBookRatioTTM', 'priceToEarningsGrowthRatioTTM', 'priceToEarningsRatioTTM', 'priceToFairValueTTM', 'priceToFreeCashFlowRatioTTM', 'priceToOperatingCashFlowRatioTTM', 'priceToSalesRatioTTM', 'quickRatioTTM', 'receivablesTurnoverTTM', 'revenuePerShareTTM', 'shareholdersEquityPerShareTTM', 'shortTermOperatingCashFlowCoverageRatioTTM', 'solvencyRatioTTM', 'tangibleBookValuePerShareTTM', 'workingCapitalTurnoverRatioTTM']
        },
        'key_metrics_ttm': {
            'description': 'TTM Key Metrics: marketCapTTM, peRatioTTM, pbRatioTTM, roeTTM, roicTTM, etc. Uses date for time series.',
            'primary_sort_column': 'marketCapTTM', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'date', 
            'period_column': None,
            'key_columns': ['symbol', 'marketCapTTM', 'pbRatioTTM', 'roeTTM', 'roicTTM', 'date'],
            'filter_hints': 'For latest data, use ORDER BY date DESC. TTM metrics are current as of the date field.',
            'is_time_series': True,
            'special_notes': 'TTM metrics are current performance metrics. Uses date field for time series.',
            'actual_columns': ['symbol', 'date', 'pbRatioTTM', 'debtToEquityTTM', 'averageInventoryTTM', 'averagePayablesTTM', 'averageReceivablesTTM', 'bookValuePerShareTTM', 'capexPerShareTTM', 'capexToDepreciationTTM', 'capexToOperatingCashFlowTTM', 'capexToRevenueTTM', 'cashPerShareTTM', 'currentRatioTTM', 'daysOfInventoryOnHandTTM', 'daysPayablesOutstandingTTM', 'daysSalesOutstandingTTM', 'debtToAssetsTTM', 'dividendYieldTTM', 'earningsYieldTTM', 'enterpriseValueTTM', 'enterpriseValueOverEBITDATTM', 'evToFreeCashFlowTTM', 'evToOperatingCashFlowTTM', 'evToSalesTTM', 'freeCashFlowPerShareTTM', 'freeCashFlowYieldTTM', 'grahamNetNetTTM', 'grahamNumberTTM', 'incomeQualityTTM', 'intangiblesToTotalAssetsTTM', 'interestCoverageTTM', 'interestDebtPerShareTTM', 'inventoryTurnoverTTM', 'investedCapitalTTM', 'marketCapTTM', 'netCurrentAssetValueTTM', 'netDebtToEBITDATTM', 'netIncomePerShareTTM', 'operatingCashFlowPerShareTTM', 'payablesTurnoverTTM', 'payoutRatioTTM', 'pfcfRatioTTM', 'pocfratioTTM', 'priceToSalesRatioTTM', 'ptbRatioTTM', 'receivablesTurnoverTTM', 'revenuePerShareTTM', 'roeTTM', 'roicTTM', 'salesGeneralAndAdministrativeToRevenueTTM', 'shareholdersEquityPerShareTTM', 'stockBasedCompensationToRevenueTTM', 'tangibleAssetValueTTM', 'tangibleBookValuePerShareTTM', 'workingCapitalTTM']
        },
        
        # Historical Market Cap
        'historical_market_cap': {
            'description': 'Historical daily market capitalization data: symbol, date, marketCap. Simple time series data.',
            'primary_sort_column': 'marketCap', 
            'sort_direction': 'DESC',
            'date_columns': ['date'],
            'time_column': 'date', 
            'period_column': None,
            'key_columns': ['symbol', 'date', 'marketCap'],
            'filter_hints': 'Use date for time filtering. Extract year with EXTRACT(year FROM date).',
            'is_time_series': True,
            'special_notes': 'Simple time series with just symbol, date, and marketCap columns.',
            'actual_columns': ['symbol', 'date', 'marketCap']
        },
                                
        # Growth Rates Tables (created from TTM data)
        'income_statements_growth_rates': {
            'description': 'Year-over-Year growth rates for income statement metrics: revenue, netIncome, eps, etc. Calculated from TTM data comparing latest vs previous year.',
            'primary_sort_column': 'revenue_yoy_growth_pct', 
            'sort_direction': 'DESC',
            'date_columns': ['latest_date', 'previous_date'],
            'time_column': 'latest_date', 
            'period_column': 'latest_period',
            'key_columns': ['symbol', 'latest_date', 'previous_date', 'revenue_yoy_growth_pct', 'netIncome_yoy_growth_pct', 'eps_yoy_growth_pct'],
            'filter_hints': 'Use latest_date for time filtering. For highest growth, use ORDER BY revenue_yoy_growth_pct DESC.',
            'is_time_series': True,
            'special_notes': 'YoY growth rates calculated from TTM data. Each row compares latest period vs same period previous year.',
            'actual_columns': [
                'symbol', 'latest_date', 'previous_date', 'days_between_periods',
                'latest_fiscal_year', 'previous_fiscal_year', 'latest_period', 'previous_period',
                'data_type', 'created_at',
                # Revenue metrics
                'revenue_current', 'revenue_previous', 'revenue_yoy_growth_pct', 'revenue_yoy_growth_amount',
                'costOfRevenue_current', 'costOfRevenue_previous', 'costOfRevenue_yoy_growth_pct', 'costOfRevenue_yoy_growth_amount',
                'grossProfit_current', 'grossProfit_previous', 'grossProfit_yoy_growth_pct', 'grossProfit_yoy_growth_amount',
                # Income metrics
                'operatingIncome_current', 'operatingIncome_previous', 'operatingIncome_yoy_growth_pct', 'operatingIncome_yoy_growth_amount',
                'netIncome_current', 'netIncome_previous', 'netIncome_yoy_growth_pct', 'netIncome_yoy_growth_amount',
                # Per-share metrics
                'eps_current', 'eps_previous', 'eps_yoy_growth_pct', 'eps_yoy_growth_amount',
                'epsDiluted_current', 'epsDiluted_previous', 'epsDiluted_yoy_growth_pct', 'epsDiluted_yoy_growth_amount',
                # Additional metrics
                'ebitda_current', 'ebitda_previous', 'ebitda_yoy_growth_pct', 'ebitda_yoy_growth_amount',
                'ebit_current', 'ebit_previous', 'ebit_yoy_growth_pct', 'ebit_yoy_growth_amount'
            ]
        },
        'balance_sheets_growth_rates': {
            'description': 'Year-over-Year growth rates for balance sheet metrics: totalAssets, totalEquity, cash, debt, etc. Calculated from TTM data comparing latest vs previous year.',
            'primary_sort_column': 'totalAssets_yoy_growth_pct', 
            'sort_direction': 'DESC',
            'date_columns': ['latest_date', 'previous_date'],
            'time_column': 'latest_date', 
            'period_column': 'latest_period',
            'key_columns': ['symbol', 'latest_date', 'previous_date', 'totalAssets_yoy_growth_pct', 'totalEquity_yoy_growth_pct', 'cashAndCashEquivalents_yoy_growth_pct'],
            'filter_hints': 'Use latest_date for time filtering. For highest growth, use ORDER BY totalAssets_yoy_growth_pct DESC.',
            'is_time_series': True,
            'special_notes': 'YoY growth rates calculated from TTM data. Each row compares latest period vs same period previous year.',
            'actual_columns': [
                'symbol', 'latest_date', 'previous_date', 'days_between_periods',
                'latest_fiscal_year', 'previous_fiscal_year', 'latest_period', 'previous_period',
                'data_type', 'created_at',
                # Asset metrics
                'totalAssets_current', 'totalAssets_previous', 'totalAssets_yoy_growth_pct', 'totalAssets_yoy_growth_amount',
                'totalCurrentAssets_current', 'totalCurrentAssets_previous', 'totalCurrentAssets_yoy_growth_pct', 'totalCurrentAssets_yoy_growth_amount',
                'cashAndCashEquivalents_current', 'cashAndCashEquivalents_previous', 'cashAndCashEquivalents_yoy_growth_pct', 'cashAndCashEquivalents_yoy_growth_amount',
                'accountsReceivables_current', 'accountsReceivables_previous', 'accountsReceivables_yoy_growth_pct', 'accountsReceivables_yoy_growth_amount',
                'inventory_current', 'inventory_previous', 'inventory_yoy_growth_pct', 'inventory_yoy_growth_amount',
                'propertyPlantEquipmentNet_current', 'propertyPlantEquipmentNet_previous', 'propertyPlantEquipmentNet_yoy_growth_pct', 'propertyPlantEquipmentNet_yoy_growth_amount',
                'goodwill_current', 'goodwill_previous', 'goodwill_yoy_growth_pct', 'goodwill_yoy_growth_amount',
                'intangibleAssets_current', 'intangibleAssets_previous', 'intangibleAssets_yoy_growth_pct', 'intangibleAssets_yoy_growth_amount',
                # Liability and equity metrics
                'totalLiabilities_current', 'totalLiabilities_previous', 'totalLiabilities_yoy_growth_pct', 'totalLiabilities_yoy_growth_amount',
                'totalCurrentLiabilities_current', 'totalCurrentLiabilities_previous', 'totalCurrentLiabilities_yoy_growth_pct', 'totalCurrentLiabilities_yoy_growth_amount',
                'totalEquity_current', 'totalEquity_previous', 'totalEquity_yoy_growth_pct', 'totalEquity_yoy_growth_amount',
                'totalStockholdersEquity_current', 'totalStockholdersEquity_previous', 'totalStockholdersEquity_yoy_growth_pct', 'totalStockholdersEquity_yoy_growth_amount',
                'retainedEarnings_current', 'retainedEarnings_previous', 'retainedEarnings_yoy_growth_pct', 'retainedEarnings_yoy_growth_amount',
                'commonStock_current', 'commonStock_previous', 'commonStock_yoy_growth_pct', 'commonStock_yoy_growth_amount',
                'additionalPaidInCapital_current', 'additionalPaidInCapital_previous', 'additionalPaidInCapital_yoy_growth_pct', 'additionalPaidInCapital_yoy_growth_amount',
                'treasuryStock_current', 'treasuryStock_previous', 'treasuryStock_yoy_growth_pct', 'treasuryStock_yoy_growth_amount',
                # Debt metrics
                'totalDebt_current', 'totalDebt_previous', 'totalDebt_yoy_growth_pct', 'totalDebt_yoy_growth_amount',
                'netDebt_current', 'netDebt_previous', 'netDebt_yoy_growth_pct', 'netDebt_yoy_growth_amount',
                'longTermDebt_current', 'longTermDebt_previous', 'longTermDebt_yoy_growth_pct', 'longTermDebt_yoy_growth_amount',
                'shortTermDebt_current', 'shortTermDebt_previous', 'shortTermDebt_yoy_growth_pct', 'shortTermDebt_yoy_growth_amount'
            ]
        },
        'cash_flow_statements_growth_rates': {
            'description': 'Year-over-Year growth rates for cash flow metrics: operatingCashFlow, freeCashFlow, capitalExpenditure, etc. Calculated from TTM data comparing latest vs previous year.',
            'primary_sort_column': 'operatingCashFlow_yoy_growth_pct', 
            'sort_direction': 'DESC',
            'date_columns': ['latest_date', 'previous_date'],
            'time_column': 'latest_date', 
            'period_column': 'latest_period',
            'key_columns': ['symbol', 'latest_date', 'previous_date', 'operatingCashFlow_yoy_growth_pct', 'freeCashFlow_yoy_growth_pct', 'capitalExpenditure_yoy_growth_pct'],

            'filter_hints': 'Use latest_date for time filtering. For highest growth, use ORDER BY operatingCashFlow_yoy_growth_pct DESC.',
            'is_time_series': True,
            'special_notes': 'YoY growth rates calculated from TTM data. Each row compares latest period vs same period previous year.',
            'actual_columns': [
                'symbol', 'latest_date', 'previous_date', 'days_between_periods',
                'latest_fiscal_year', 'previous_fiscal_year', 'latest_period', 'previous_period',
                'data_type', 'created_at',
                # Operating cash flow metrics
                'operatingCashFlow_current', 'operatingCashFlow_previous', 'operatingCashFlow_yoy_growth_pct', 'operatingCashFlow_yoy_growth_amount',
                'netCashProvidedByOperatingActivities_current', 'netCashProvidedByOperatingActivities_previous', 'netCashProvidedByOperatingActivities_yoy_growth_pct', 'netCashProvidedByOperatingActivities_yoy_growth_amount',
                'freeCashFlow_current', 'freeCashFlow_previous', 'freeCashFlow_yoy_growth_pct', 'freeCashFlow_yoy_growth_amount',
                # Investing cash flow metrics
                'capitalExpenditure_current', 'capitalExpenditure_previous', 'capitalExpenditure_yoy_growth_pct', 'capitalExpenditure_yoy_growth_amount',
                'purchasesOfInvestments_current', 'purchasesOfInvestments_previous', 'purchasesOfInvestments_yoy_growth_pct', 'purchasesOfInvestments_yoy_growth_amount',
                'salesMaturitiesOfInvestments_current', 'salesMaturitiesOfInvestments_previous', 'salesMaturitiesOfInvestments_yoy_growth_pct', 'salesMaturitiesOfInvestments_yoy_growth_amount',
                'acquisitionsNet_current', 'acquisitionsNet_previous', 'acquisitionsNet_yoy_growth_pct', 'acquisitionsNet_yoy_growth_amount',
                # Financing cash flow metrics
                'netCashProvidedByInvestingActivities_current', 'netCashProvidedByInvestingActivities_previous', 'netCashProvidedByInvestingActivities_yoy_growth_pct', 'netCashProvidedByInvestingActivities_yoy_growth_amount',
                'netCashProvidedByFinancingActivities_current', 'netCashProvidedByFinancingActivities_previous', 'netCashProvidedByFinancingActivities_yoy_growth_pct', 'netCashProvidedByFinancingActivities_yoy_growth_amount',
                # Additional metrics
                'changeInWorkingCapital_current', 'changeInWorkingCapital_previous', 'changeInWorkingCapital_yoy_growth_pct', 'changeInWorkingCapital_yoy_growth_amount',
                'depreciationAndAmortization_current', 'depreciationAndAmortization_previous', 'depreciationAndAmortization_yoy_growth_pct', 'depreciationAndAmortization_yoy_growth_amount',
                'stockBasedCompensation_current', 'stockBasedCompensation_previous', 'stockBasedCompensation_yoy_growth_pct', 'stockBasedCompensation_yoy_growth_amount',
                'commonDividendsPaid_current', 'commonDividendsPaid_previous', 'commonDividendsPaid_yoy_growth_pct', 'commonDividendsPaid_yoy_growth_amount',
                'commonStockRepurchased_current', 'commonStockRepurchased_previous', 'commonStockRepurchased_yoy_growth_pct', 'commonStockRepurchased_yoy_growth_amount',
                'commonStockIssuance_current', 'commonStockIssuance_previous', 'commonStockIssuance_yoy_growth_pct', 'commonStockIssuance_yoy_growth_amount'
            ]
        }
    }
