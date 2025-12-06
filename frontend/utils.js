// INDEPENDENT UTILS - No global dependencies
// All functions accept data as parameters instead of accessing global variables

// Enhanced column detection that includes year columns
function getColumnsByType(type, dataContext) {
    const { 
        currentMultiSheetData, 
        currentSheets, 
        lastApiResponse, 
        chartSheetSelectorValue 
    } = dataContext;
    
    let dataSource, columnsSource;
    
    // Determine data source (single sheet vs multi-sheet vs combined)
    if (currentMultiSheetData && chartSheetSelectorValue === 'combined') {
        // Combined analysis - get columns from all successful sheets
        const successfulSheets = currentSheets.filter(sheet => sheet.success !== false && sheet.data && sheet.data.length > 0);
        if (successfulSheets.length === 0) return [];
        
        // Use columns from first successful sheet and validate across all
        const baseColumns = successfulSheets[0].columns || [];
        columnsSource = baseColumns.filter(col => 
            successfulSheets.every(sheet => sheet.columns && sheet.columns.includes(col))
        );
        
        // Sample data from first sheet for type detection
        dataSource = successfulSheets[0].data;
    } else if (currentMultiSheetData && chartSheetSelectorValue !== '') {
        const sheetIndex = parseInt(chartSheetSelectorValue);
        const sheet = currentSheets[sheetIndex];
        if (!sheet || sheet.success === false || !sheet.data || sheet.data.length === 0) {
            return [];
        }
        dataSource = sheet.data;
        columnsSource = sheet.columns || [];
    } else if (lastApiResponse && lastApiResponse.data_rows) {
        dataSource = lastApiResponse.data_rows;
        columnsSource = lastApiResponse.columns || [];
    } else {
        return [];
    }

    // Helper to normalize column names for time exclusion
    function normalizeColName(colKey) {
        return colKey.replace(/_/g, '').toLowerCase();
    }
    const timeColNames = [
        'year', 'date', 'period', 'calendaryear', 'fiscalyear'
    ];

    return columnsSource.filter(colKey => {
        const colLower = colKey.toLowerCase();
        const normCol = normalizeColName(colKey);
        // Check if this is a year/date/period column (for exclusion from numeric)
        const isExplicitTimeCol = timeColNames.includes(normCol);
        // For categorical: keep existing logic
        if (type === 'categorical') {
            // For categorical: include year, date, and text columns
            const isYearColumn = colLower.includes('year') || colLower.includes('calendaryear') || 
                                colLower.includes('fiscal_year') || colLower === 'yr';
            const isDateColumn = colLower.includes('date') || colLower.includes('time') || colLower.includes('period');
            if (isYearColumn || isDateColumn) return true;
            for (let i = 0; i < Math.min(dataSource.length, 5); i++) {
                const value = dataSource[i][colKey];
                if (value !== null && value !== undefined) {
                    const isNumeric = !isNaN(parseFinancialNumber(value));
                    if (!isNumeric) return true;
                }
            }
            return false;
        } else {
            // For numeric: only exclude if column name is exactly a time column
            if (isExplicitTimeCol) return false;
            // Check up to 20 non-null values (or all if fewer)
            let checked = 0, numeric = 0;
            for (let i = 0; i < dataSource.length && checked < 20; i++) {
                const value = dataSource[i][colKey];
                if (value !== null && value !== undefined) {
                    checked++;
                    if (!isNaN(parseFinancialNumber(value))) numeric++;
                }
            }
            // Consider numeric if at least 80% of checked non-null values are numbers
            return checked > 0 && (numeric / checked) >= 0.8;
        }
    });
}

// Helper function to parse financial numbers
function parseFinancialNumber(value) {
    if (value === null || value === undefined) return NaN;
    if (typeof value === 'number') return value;

    let str = String(value).trim().toUpperCase();
    const multipliers = { 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12 };
    const suffix = str.slice(-1);
    
    let multiplier = 1;
    if (multipliers[suffix]) {
        multiplier = multipliers[suffix];
        str = str.slice(0, -1);
    }
    
    str = str.replace(/[\$,%]/g, '');
    const num = parseFloat(str);

    return isNaN(num) ? NaN : num * multiplier;
}

// Format column name helper
function formatColumnName(colKey) {
    return colKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Check if data represents time series
function isTimeSeriesData(dataSource, xCol) {
    if (!xCol || !dataSource.length) return false;
    
    const isTimeColumn = xCol.toLowerCase().includes('year') || 
                       xCol.toLowerCase().includes('calendaryear') ||
                       xCol.toLowerCase().includes('date') ||
                       xCol.toLowerCase().includes('time') ||
                       xCol.toLowerCase().includes('period');
    
    if (!isTimeColumn) return false;
    
    // Check if we have multiple time points
    const uniqueTimeValues = new Set(dataSource.map(row => row[xCol]));
    return uniqueTimeValues.size > 1;
}

// Check for multiple series in data
function hasMultipleSeries(dataSource, xCol) {
    if (!dataSource.length) return { hasMultiple: false, seriesColumn: null, values: [] };
    
    // Look for segment/company/category/geography columns
    const possibleSeriesColumns = [
        'segment_name', 'segment', 'geography_segment', 'Geography_Segment', 'geography', 'Geography',
        'company', 'companyName', '_company', 'category', 'type', 'region', 'Region'
    ];
    
    for (const seriesCol of possibleSeriesColumns) {
        if (dataSource[0].hasOwnProperty(seriesCol)) {
            const uniqueValues = new Set(dataSource.map(row => row[seriesCol]).filter(val => val !== null && val !== undefined && val !== ''));
            if (uniqueValues.size > 1) {
                return { hasMultiple: true, seriesColumn: seriesCol, values: [...uniqueValues] };
            }
        }
    }
    
    return { hasMultiple: false, seriesColumn: null, values: [] };
}

// Format financial values
function formatFinancialValue(value, colKey, compactNumber = false, format = null, sampleValues = null, friendlyColumns = {}) {
    if (value === null || value === undefined) return '—';
    const num = parseFinancialNumber(value);
    if(isNaN(num)) return value;
    
    // Auto-detect format if not provided
    if (!format) {
        format = detectColumnFormat(colKey, sampleValues, friendlyColumns);
    }
    
    // Format based on detected type
    switch (format) {
        case 'percentage':
            return num.toFixed(2) + '%';
            
        case 'ratio':
            return num.toFixed(2);
            
        case 'year':
            return Math.round(num).toString();
            
        case 'count':
            return num.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            
        case 'currency':
            return formatCurrency(num, colKey);
            
        default:
            // Always use two decimals for compact formatting
            if (Math.abs(num) >= 1e12) {
                return `${(num / 1e12).toFixed(2)}T`;
            } else if (Math.abs(num) >= 1e9) {
                return `${(num / 1e9).toFixed(2)}B`;
            } else if (Math.abs(num) >= 1e6) {
                return `${(num / 1e6).toFixed(2)}M`;
            } else if (Math.abs(num) >= 1e3) {
                return `${(num / 1e3).toFixed(2)}K`;
            } else {
                return num.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }
    }
}

// Detect column format
function detectColumnFormat(colKey, sampleValues, friendlyColumns = {}) {
    const key = colKey ? colKey.toLowerCase() : '';
    const friendlyName = friendlyColumns[colKey] ? friendlyColumns[colKey].toLowerCase() : '';
    
    // Check column name patterns for format hints
    const formatPatterns = {
        percentage: [
            // Explicit percentage terms only
            /\bpercent(age)?\b/i, /\bpct\b/i, /%/i, 
            // Growth rates (but not profit growth)
            /\bcagr\b/i, /\bgrowth\s*rate\b/i, /\byield\b/i,
            // Specific margin types (but exclude gross profit, net profit, etc.)
            /\bmargin(?!\s*(call|loan|trading))\b/i,
            // Specific financial ratios that are percentages
            /\broe\b/i, /\broa\b/i, /\broic\b/i, /\broe\b/i,
            // Tax and interest rates
            /\btax\s*rate\b/i, /\binterest\s*rate\b/i,
            // Change percentages
            /\bchange\s*%\b/i, /\b%\s*change\b/i
        ],
        ratio: [
            // Specific valuation ratios
            /\bp\/e\b/i, /\bpe[\s_]ratio\b/i, /\bp\/b\b/i, /\bpb[\s_]ratio\b/i,
            /\bprice[\s_]to[\s_]earnings\b/i, /\bprice[\s_]to[\s_]book\b/i,
            /\bprice[\s_]to[\s_]sales\b/i, /\bp\/s\b/i,
            /\bev\/ebitda\b/i, /\bev\/revenue\b/i,
            // Debt ratios
            /\bdebt[\s_]to[\s_]equity\b/i, /\bdebt[\s_]to[\s_]assets\b/i,
            /\bcurrent[\s_]ratio\b/i, /\bquick[\s_]ratio\b/i,
            // Coverage ratios
            /\binterest[\s_]coverage\b/i, /\bcoverage[\s_]ratio\b/i,
            // Turnover ratios
            /\bturnover[\s_]ratio\b/i, /\basset[\s_]turnover\b/i,
            // General ratio (but be more specific)
            /\bratio(?!\s*(analysis|calculation))\b/i
        ],
        currency: [
            // Only specific financial amounts that should have $ symbols
            /\brevenue\b/i, /\bincome\b/i, /\bcash\b/i,
            // Specific profit terms (these are currency, not percentages!)
            /\bgross[\s_]profit\b/i, /\bnet[\s_]profit\b/i, /\boperating[\s_]profit\b/i,
            /\bebitda\b/i, /\bebit\b/i,
            // Assets and liabilities (but not ratios)
            /\bassets\b/i, /\bliabilities\b/i, /\bequity\b/i, /\bdebt\b/i,
            // Expenses and R&D
            /\bexpenses?\b/i, /\bcapex\b/i, /\bopex\b/i, /\brnd\b/i, /\bresearch\b/i, /\bdevelopment\b/i,
            // Market cap and price (but not P/E, P/B ratios)
            /\bmarket[\s_]cap\b/i, /\bprice(?![\s_]to)/i,
            // Explicit currency indicators
            /\$/, /\busd\b/i, /\bamount\b/i, /\bfee\b/i
        ],
        count: [
            /\bcount\b/i, /\bnumber\b/i, /\bqty\b/i, /\bquantity\b/i, 
            /\bshares\b/i, /\bemployees\b/i, /\boutstanding\b/i
        ],
        year: [
            /\byear\b/i, /\byr\b/i, /\bdate\b/i, /\btime\b/i, /\bperiod\b/i,
            /\bcalendar[\s_]year\b/i, /\bfiscal[\s_]year\b/i
        ]
    };
    
    // Check column name against patterns with priority order
    // 1. Check for ratio patterns FIRST (highest priority - never use $ with ratios)
    for (const pattern of formatPatterns.ratio) {
        if (pattern.test(key) || pattern.test(friendlyName)) {
            return 'ratio';
        }
    }
    
    // 2. Check for year patterns
    for (const pattern of formatPatterns.year) {
        if (pattern.test(key) || pattern.test(friendlyName)) {
            return 'year';
        }
    }
    
    // 3. Check for count patterns
    for (const pattern of formatPatterns.count) {
        if (pattern.test(key) || pattern.test(friendlyName)) {
            return 'count';
        }
    }
    
    // 4. Check for percentage patterns
    for (const pattern of formatPatterns.percentage) {
        if (pattern.test(key) || pattern.test(friendlyName)) {
            return 'percentage';
        }
    }
    
    // 5. Check for currency patterns LAST (only for specific financial amounts)
    for (const pattern of formatPatterns.currency) {
        if (pattern.test(key) || pattern.test(friendlyName)) {
            return 'currency';
        }
    }
    
    // Analyze sample values to detect format
    if (sampleValues && sampleValues.length > 0) {
        const numericValues = sampleValues
            .map(v => parseFinancialNumber(v))
            .filter(v => !isNaN(v));
        
        if (numericValues.length > 0) {
            const avgValue = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
            const maxValue = Math.max(...numericValues);
            const minValue = Math.min(...numericValues);
            
            // If values are years (1900-2100)
            if (minValue >= 1900 && maxValue <= 2100) {
                return 'year';
            }
            
            // If values are small and between -100 and 100, might be percentage
            if (maxValue <= 100 && minValue >= -100 && Math.abs(avgValue) < 50) {
                // But only if column name suggests percentage
                if (formatPatterns.percentage.some(pattern => pattern.test(key) || pattern.test(friendlyName))) {
                    return 'percentage';
                }
            }
            
            // If values are small positive numbers (< 1000), might be ratios
            if (maxValue < 1000 && minValue >= 0 && avgValue < 100) {
                // But only if column name suggests ratio
                if (formatPatterns.ratio.some(pattern => pattern.test(key) || pattern.test(friendlyName))) {
                    return 'ratio';
                }
            }
            
            // Large values might be currency - but be very selective
            if (maxValue > 1000 || avgValue > 100) {
                // Only use currency format for specific financial amounts, not ratios
                if (formatPatterns.currency.some(pattern => pattern.test(key) || pattern.test(friendlyName)) &&
                    !formatPatterns.ratio.some(pattern => pattern.test(key) || pattern.test(friendlyName))) {
                    return 'currency';
                }
            }
        }
    }
    
    // NO DEFAULT FALLBACK - return null for unrecognized columns
    return null;
}

// Get sector class for styling
function getSectorClass(sectorName) {
    if (!sectorName) return 'sector-default';
    const s = sectorName.toLowerCase();
    if (s.includes('tech')) return 'sector-technology';
    if (s.includes('health')) return 'sector-healthcare';
    if (s.includes('financ')) return 'sector-financials';
    return 'sector-default';
}

// Check if a column is filing-related and should be handled separately
function isFilingColumn(columnName) {
    const lowerCol = columnName.toLowerCase();
    
    // Remove year suffixes like _2025, _2024, _2023, etc. to check the base column name
    const baseCol = lowerCol.replace(/_\d{4}$/, '');
    
    // Exclude relevance and reasoning columns from being treated as filing columns
    if (baseCol === 'relevance' || baseCol === 'reasoning') {
        return false;
    }
    
    return baseCol.includes('filing') || 
           baseCol.includes('link') || 
           baseCol.includes('cik') || 
           baseCol.includes('accepted') || 
           baseCol.includes('filling') ||
           baseCol === 'period' || // period column by itself is often filing-related
           (baseCol.includes('period') && (baseCol.includes('report') || baseCol.includes('filing')));
}

// Filter columns to separate data columns from filing columns
function separateDataAndFilingColumns(columns, friendlyColumns = {}) {
    const dataColumns = [];
    const filingColumns = [];
    const dataFriendlyColumns = {};
    const filingFriendlyColumns = {};
    
    columns.forEach(col => {
        if (isFilingColumn(col)) {
            filingColumns.push(col);
            if (friendlyColumns[col]) {
                filingFriendlyColumns[col] = friendlyColumns[col];
            }
        } else {
            dataColumns.push(col);
            if (friendlyColumns[col]) {
                dataFriendlyColumns[col] = friendlyColumns[col];
            }
        }
    });
    
    return { dataColumns, filingColumns, dataFriendlyColumns, filingFriendlyColumns };
}

// Extract filing data from result (for SEC transparency features)
function extractFilingData(result) {
    
    const filingData = [];
    
    if (result.query_type === 'multi_sheet' && result.sheets) {
        // Multi-sheet filing data extraction
        result.sheets.forEach((sheet, sheetIndex) => {
            // Check if backend provided separated filing data
            if (sheet.filing_data && sheet.filing_data.length > 0) {
                filingData.push({
                    company: sheet.company,
                    sheetIndex: sheetIndex,
                    filings: sheet.filing_data
                });
            } else if (sheet.data && sheet.data.length > 0) {
                // Fallback to old extraction method
                const sheetFilings = extractFilingFromSheetData(sheet.data, sheet.company);
                if (sheetFilings.length > 0) {
                    filingData.push({
                        company: sheet.company,
                        sheetIndex: sheetIndex,
                        filings: sheetFilings
                    });
                }
            }
        });
    } else if (result.filing_data && result.filing_data.length > 0) {
        // NEW: Use backend-provided separated filing data for single-sheet
        filingData.push({
            company: null,
            sheetIndex: null,
            filings: result.filing_data
        });
    } else if (result.data_rows && result.data_rows.length > 0) {
        // Fallback: Single-sheet filing data extraction using old method
        const singleSheetFilings = extractFilingFromSheetData(result.data_rows, null);
        if (singleSheetFilings.length > 0) {
            filingData.push({
                company: null,
                sheetIndex: null,
                filings: singleSheetFilings
            });
        }
    }
    
    return filingData;
}

// Extract filing information from sheet data
function extractFilingFromSheetData(dataRows, company) {
    const filings = [];
    const seenFilings = new Set();
    
    dataRows.forEach(row => {
        const filing = {};
        let hasFilingInfo = false;
        
        // Extract filing-related columns
        Object.keys(row).forEach(key => {
            const lowerKey = key.toLowerCase();
            if (lowerKey.includes('link') || lowerKey.includes('filling') || 
                lowerKey.includes('accepted') || lowerKey.includes('cik') ||
                lowerKey.includes('filing')) {
                filing[key] = row[key];
                if (row[key] && row[key] !== '-' && row[key] !== 'No filing link available') {
                    hasFilingInfo = true;
                }
            }
        });
        
        // Add company context if available
        if (row.symbol) filing.symbol = row.symbol;
        if (row.companyName) filing.companyName = row.companyName;
        if (company) filing.company = company;
        
        // Only add if we have actual filing data and haven't seen this exact filing
        if (hasFilingInfo) {
            const filingKey = `${filing.symbol || company || 'unknown'}_${filing.link || filing.fillingDate || JSON.stringify(filing)}`;
            if (!seenFilings.has(filingKey)) {
                seenFilings.add(filingKey);
                filings.push(filing);
            }
        }
    });
    
    return filings;
}

// Generate highly distinct color palette for charts
function generateColorPalette(count) {
    // Carefully curated colors for maximum distinction and accessibility
    const highContrastColors = [
        '#1f77b4', // Strong Blue
        '#ff7f0e', // Vivid Orange  
        '#2ca02c', // Forest Green
        '#d62728', // Crimson Red
        '#9467bd', // Purple
        '#8c564b', // Brown
        '#e377c2', // Pink
        '#7f7f7f', // Gray
        '#bcbd22', // Olive
        '#17becf', // Cyan
        '#ff9896', // Light Red
        '#c5b0d5', // Light Purple
        '#c49c94', // Light Brown
        '#f7b6d3', // Light Pink
        '#dbdb8d', // Light Olive
        '#9edae5', // Light Cyan
        '#ffbb78', // Light Orange
        '#98df8a', // Light Green
        '#aec7e8', // Light Blue
        '#c7c7c7', // Light Gray
    ];
    
    if (count <= highContrastColors.length) {
        return highContrastColors.slice(0, count);
    }
    
    // For additional colors beyond the base set, use advanced color theory
    const colors = [...highContrastColors];
    const baseHues = [0, 30, 60, 120, 180, 210, 240, 270, 300, 330]; // Well-spaced hues
    
    for (let i = highContrastColors.length; i < count; i++) {
        // Use golden ratio for optimal color spacing
        const goldenRatio = 0.618033988749895;
        const hueIndex = Math.floor(i / 3) % baseHues.length;
        const baseHue = baseHues[hueIndex];
        
        // Vary saturation and lightness for distinction
        const variation = i % 3;
        let saturation, lightness;
        
        switch (variation) {
            case 0: // Vibrant
                saturation = 80;
                lightness = 50;
                break;
            case 1: // Medium
                saturation = 65;
                lightness = 65;
                break;
            case 2: // Muted
                saturation = 45;
                lightness = 55;
                break;
        }
        
        // Add some randomness to hue for better distribution
        const hue = (baseHue + (i * goldenRatio * 60)) % 360;
        
        // Convert HSL to RGB then to hex
        const h = hue / 360;
        const s = saturation / 100;
        const l = lightness / 100;
        
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        let r, g, b;
        if (s === 0) {
            r = g = b = l;
        } else {
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        
        // Convert to hex
        const toHex = (c) => {
            const hex = Math.round(c * 255).toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        };
        
        const hexColor = `#${toHex(r)}${toHex(g)}${toHex(b)}`;
        colors.push(hexColor);
    }
    
    return colors;
}

// ===== FORMATTING UTILITIES =====

/**
 * Format currency values with appropriate suffixes (K, M, B, T)
 * @param {number} value - The value to format
 * @returns {string|null} Formatted currency string or null if invalid
 */
function formatCurrency(value, columnName = '') {
    if (!value || isNaN(value)) return null;
    const isStrictCurrency = columnName && (
        columnName.toLowerCase().includes('revenue') ||
        columnName.toLowerCase().includes('income') ||
        columnName.toLowerCase().includes('marketcap') ||
        columnName.toLowerCase().includes('mktcap') ||
        columnName.toLowerCase().includes('price') ||
        columnName.toLowerCase().includes('sales')
    );
    const num = Math.abs(value);
    let formattedValue;
    if (num >= 1e12) {
        formattedValue = `${(value / 1e12).toFixed(2)}T`;
    } else if (num >= 1e9) {
        formattedValue = `${(value / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        formattedValue = `${(value / 1e6).toFixed(2)}M`;
    } else {
        formattedValue = new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
    return formattedValue;
}

/**
 * Format numbers with appropriate suffixes (K, M, B, T)
 * @param {number} value - The value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string|null} Formatted number string or null if invalid
 */
function formatNumber(value, decimals = 0) {
    if (!value || isNaN(value)) return null;
    
    const num = Math.abs(value);
    
    if (num >= 1e12) {
        return `${(value / 1e12).toFixed(2)}T`;
    } else if (num >= 1e9) {
        return `${(value / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        return `${(value / 1e6).toFixed(2)}M`;
    } else {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value);
    }
}

/**
 * Format percentage values - Return raw value (backend no longer converts percentages)
 * @param {number} value - The value to format (raw value from backend)
 * @returns {string|null} Raw value string or null if invalid
 */
function formatPercentage(value) {
    if (!value || isNaN(value)) return null;
    
    // Format percentage with two decimal places
    const num = parseFloat(value);
    return num.toFixed(2) + '%';
}

/**
 * Format range values (low - high)
 * @param {number} low - Lower bound
 * @param {number} high - Upper bound
 * @returns {string|null} Formatted range string or null if invalid
 */
function formatRange(low, high) {
    if (!low || !high || isNaN(low) || isNaN(high)) return null;
    return `${formatCurrency(low, '')} - ${formatCurrency(high, '')}`;
}

/**
 * Format range from a string like "169.21-260.10"
 * @param {string} rangeStr - Range string to parse
 * @returns {string|null} Formatted range string or original string if parsing fails
 */
function formatRangeFromString(rangeStr) {
    if (!rangeStr) return null;
    try {
        const parts = rangeStr.toString().split('-');
        if (parts.length === 2) {
            const low = parseFloat(parts[0].trim());
            const high = parseFloat(parts[1].trim());
            if (!isNaN(low) && !isNaN(high)) {
                return formatRange(low, high);
            }
        }
    } catch (e) {
        // Fall back to original string if parsing fails
    }
    return rangeStr;
}

/**
 * Format change values with + sign for positive values
 * @param {number} value - The change value
 * @returns {string|null} Formatted change string or null if invalid
 */
function formatChange(value) {
    if (!value || isNaN(value)) return null;
    const sign = value >= 0 ? '+' : '';
    return `${sign}${formatCurrency(value, '')}`;
}

/**
 * Get CSS class for change values (positive/negative)
 * @param {number} value - The change value
 * @returns {string} CSS class name
 */
function getChangeClass(value) {
    if (!value || isNaN(value)) return '';
    return value >= 0 ? 'positive' : 'negative';
}

/**
 * Format date strings to readable format
 * @param {string} dateString - Date string to format
 * @returns {string|null} Formatted date string or original string if parsing fails
 */
function formatDate(dateString) {
    if (!dateString) return null;
    try {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    } catch (e) {
        return dateString;
    }
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML string
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Truncate display values to specified length
 * @param {string} value - Value to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} Truncated string with ellipsis if needed
 */
function truncateDisplayValue(value, maxLength = 100) {
    if (!value) return '';
    const str = String(value);
    return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
}

/**
 * Check if a value is truncated
 * @param {string} value - Value to check
 * @param {number} maxLength - Maximum length before truncation
 * @returns {boolean} True if value is truncated
 */
function isValueTruncated(value, maxLength = 100) {
    if (!value) return false;
    const str = String(value);
    return str.length > maxLength;
}

/**
 * Get the original value before truncation
 * @param {string} value - Value to get original for
 * @returns {string} Original value
 */
function getOriginalValue(value) {
    if (!value) return '';
    return String(value);
}

/**
 * Truncate company names for display
 * @param {string} value - Company name to truncate
 * @returns {string} Truncated company name
 */
function truncateCompanyName(value) {
    if (!value) return '';
    const str = String(value);
    return str.length > 50 ? str.substring(0, 50) + '...' : str;
}

/**
 * Format market cap values with appropriate suffixes
 * @param {number} marketCap - Market cap value
 * @returns {string|null} Formatted market cap string or null if invalid
 */
function formatMarketCap(marketCap) {
    if (!marketCap || isNaN(marketCap)) return null;
    
    const num = Math.abs(marketCap);
    
    if (num >= 1e12) {
        return `$${(marketCap / 1e12).toFixed(2)}T`;
    } else if (num >= 1e9) {
        return `$${(marketCap / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        return `$${(marketCap / 1e6).toFixed(2)}M`;
    } else if (num >= 1e3) {
        return `$${(marketCap / 1e3).toFixed(2)}K`;
    } else {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(marketCap);
    }
}

/**
 * Format large numbers (shares outstanding, employee counts) with appropriate suffixes
 * @param {number} value - The large number to format
 * @returns {string|null} Formatted number string or null if invalid
 */
function formatLargeNumber(value) {
    if (!value || isNaN(value)) return null;
    
    const num = Math.abs(value);
    
    if (num >= 1e12) {
        return `${(value / 1e12).toFixed(2)}T`;
    } else if (num >= 1e9) {
        return `${(value / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        return `${(value / 1e6).toFixed(2)}M`;
    } else if (num >= 1e3) {
        return `${(value / 1e3).toFixed(2)}K`;
    } else {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    }
}

// Export all utility functions
window.ChartUtils = {
    getColumnsByType,
    parseFinancialNumber,
    formatColumnName,
    isTimeSeriesData,
    hasMultipleSeries,
    formatFinancialValue,
    detectColumnFormat,
    getSectorClass,
    isFilingColumn,
    separateDataAndFilingColumns,
    extractFilingData,
    extractFilingFromSheetData,
    generateColorPalette,
    // Add new formatting utilities
    formatCurrency,
    formatNumber,
    formatPercentage,
    formatRange,
    formatRangeFromString,
    formatChange,
    getChangeClass,
    formatDate,
    escapeHtml,
    truncateDisplayValue,
    truncateCompanyName,
    formatMarketCap,
    formatLargeNumber,
    // Add new truncation utilities
    isValueTruncated,
    getOriginalValue
};

// Make formatting functions available globally
window.formatCurrency = formatCurrency;
window.formatNumber = formatNumber;
window.formatPercentage = formatPercentage;
window.formatRange = formatRange;
window.formatRangeFromString = formatRangeFromString;
window.formatChange = formatChange;
window.getChangeClass = getChangeClass;
window.formatDate = formatDate;
window.formatMarketCap = formatMarketCap;
window.formatLargeNumber = formatLargeNumber;