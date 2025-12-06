// INDEPENDENT CHARTS - No global dependencies
// All functions accept data as parameters instead of accessing global variables

// Create standard chart (existing functionality)
function createStandardChart(dataSource, xCol, yCol, chartType, friendlyColumns) {
    const dataPointsToShow = $('#chartDataPointsSelect').val();
    
    // Handle data filtering based on chart type
    let dataForChart;
    if (dataPointsToShow === 'all') {
        dataForChart = dataSource;
    } else if (chartType === 'line') {
        // For line charts, filter by time periods (most recent years/time points)
        const yearsToShow = parseInt(dataPointsToShow) || 10;
        
        // First check if we have time-based data
        const isTimeSeriesData = dataSource.some(row => {
            const xValue = row[xCol];
            return xValue && (
                (typeof xValue === 'number' && xValue >= 1900 && xValue <= 2100) ||
                xCol.toLowerCase().includes('year') ||
                xCol.toLowerCase().includes('date')
            );
        });
        
        if (isTimeSeriesData) {
            // Sort by time column and take most recent periods
            const sortedData = [...dataSource].sort((a, b) => {
                const aTime = parseFloat(a[xCol]) || 0;
                const bTime = parseFloat(b[xCol]) || 0;
                return aTime - bTime; // Ascending order
            });
            
            // Get unique time points and take most recent
            const uniqueTimePoints = [...new Set(sortedData.map(row => row[xCol]))];
            const recentTimePoints = uniqueTimePoints.slice(-yearsToShow);
            
            // Filter data to only include recent time points
            dataForChart = sortedData.filter(row => recentTimePoints.includes(row[xCol]));
        } else {
            // If not time series, sort by Y value and take top N
            const sortedData = [...dataSource].sort((a, b) => {
                const aVal = ChartUtils.parseFinancialNumber(a[yCol]) || 0;
                const bVal = ChartUtils.parseFinancialNumber(b[yCol]) || 0;
                return bVal - aVal;
            });
            dataForChart = sortedData.slice(0, yearsToShow);
        }
    } else {
        // For bar charts, sort by Y value and take top N
        const sortedData = [...dataSource].sort((a, b) => {
            const aVal = ChartUtils.parseFinancialNumber(a[yCol]) || 0;
            const bVal = ChartUtils.parseFinancialNumber(b[yCol]) || 0;
            return bVal - aVal;
        });
        dataForChart = sortedData.slice(0, parseInt(dataPointsToShow) || 10);
    }
    
    const colors = ChartUtils.generateColorPalette(dataForChart.length);
    
    const chartData = { datasets: [] };
    chartData.labels = dataForChart.map(row => {
        // 🏢 PRIORITIZE company name for multi-company data
        const companyName = row._company || row.company || row.companyName || row.company_name;
        const xValue = row[xCol];
        
        // Use company name if available, otherwise use X column value
        let label = companyName || xValue || 'Unknown';
        
        // Clean up company names
        if (label && label.length > 12) {
            let cleanLabel = label
                .replace(/ Inc\.?$/, '')
                .replace(/ Corp\.?$/, '')
                .replace(/ Ltd\.?$/, '')
                .replace(/ LLC\.?$/, '')
                .replace(/ Co\.?$/, '');
            
            if (cleanLabel.length > 12) {
                return cleanLabel.substring(0, 10) + '..';
            }
            return cleanLabel;
        }
        return label;
    });
    
    
    if (chartType === 'bar') {
        // 🎨 BAR CHART: Use different color for each company/bar
        chartData.datasets.push({
            label: friendlyColumns[yCol] || ChartUtils.formatColumnName(yCol),
            data: dataForChart.map(row => ChartUtils.parseFinancialNumber(row[yCol])),
            backgroundColor: colors.map(c => c + '80'), // Semi-transparent fill
            borderColor: colors, // Solid border
            borderWidth: 2,
            borderRadius: 6,
            borderSkipped: false,
            // Add hover effects
            hoverBackgroundColor: colors.map(c => c + 'A0'), // More opaque on hover
            hoverBorderColor: colors.map(c => c + 'FF'), // Solid on hover
            hoverBorderWidth: 3
        });
    } else if (chartType === 'line') {
        chartData.datasets.push({
            label: friendlyColumns[yCol] || ChartUtils.formatColumnName(yCol),
            data: dataForChart.map(row => ChartUtils.parseFinancialNumber(row[yCol])),
            backgroundColor: colors[0] + '20',
            borderColor: colors[0],
            borderWidth: 3,
            fill: true,
            tension: 0.4
        });
    }
    
    return {
        data: chartData,
        options: createStandardChartOptions(xCol, yCol, chartType, friendlyColumns, dataForChart)
    };
}

// Create standard chart options function
function createStandardChartOptions(xCol, yCol, chartType, friendlyColumns, dataForChart) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
            legend: {
                display: chartType === 'line',
                position: 'top',
                labels: {
                    color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                    padding: 15,
                    usePointStyle: true,
                    font: { size: 12, family: 'Inter', weight: '600' }
                }
            },
            tooltip: {
                backgroundColor: $('html').hasClass('dark') ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                titleColor: $('html').hasClass('dark') ? '#f1f5f9' : '#0f172a',
                bodyColor: $('html').hasClass('dark') ? '#e2e8f0' : '#334155',
                borderColor: $('html').hasClass('dark') ? '#334155' : '#e2e8f0',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { size: 13, weight: 'bold', family: 'Inter' },
                bodyFont: { size: 12, family: 'Inter' },
                callbacks: {
                    title: function(context) {
                        const index = context[0].dataIndex;
                        return dataForChart[index][xCol];
                    },
                    label: (ctx) => `${ctx.dataset.label}: ${ChartUtils.formatFinancialValue(ctx.parsed.y, yCol)}`
                }
            },
            datalabels: {
                display: dataForChart.length <= 8,
                color: $('html').hasClass('dark') ? '#f1f5f9' : '#1e293b',
                font: { size: 11, weight: 'bold', family: 'Inter' },
                formatter: (value) => ChartUtils.formatFinancialValue(value, yCol, true),
                anchor: 'end',
                align: 'top',
                offset: 4
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { 
                    color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' },
                    maxRotation: 45,
                    minRotation: 20,
                    autoSkip: false,
                    callback: function(value, index, values) {
                        // Show full label if short, else truncate and add ...
                        const label = this.getLabelForValue(value);
                        if (label && label.length > 16) {
                            return label.substring(0, 14) + '…';
                        }
                        return label;
                    }
                }
            },
            y: {
                grid: { color: $('html').hasClass('dark') ? 'rgba(51, 65, 85, 0.3)' : 'rgba(226, 232, 240, 0.5)' },
                ticks: {
                    color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' },
                    callback: (value) => ChartUtils.formatFinancialValue(value, yCol, true)
                }
            }
        },
        elements: {
            point: {
                radius: chartType === 'line' ? 5 : 0,
                hoverRadius: chartType === 'line' ? 8 : 0,
                borderWidth: 2
            },
            bar: { 
                borderRadius: 6, 
                borderSkipped: false, 
                borderWidth: 2,
                // Enhanced hover effects
                hoverBorderWidth: 3,
                hoverBorderRadius: 8
            }
        },
        layout: { padding: { top: 30, right: 20, bottom: 10, left: 20 } }
    };
}

// Chart options creators for different chart types
function createTimeSeriesChartOptions(xCol, yCol, friendlyColumns) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    color: '#475569',
                    padding: 15,
                    usePointStyle: true,
                    font: { size: 12, family: 'Inter', weight: '600' }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                titleColor: '#0f172a',
                bodyColor: '#334155',
                borderColor: '#e2e8f0',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { size: 13, weight: 'bold', family: 'Inter' },
                bodyFont: { size: 12, family: 'Inter' },
                callbacks: {
                    title: (context) => `${friendlyColumns[xCol] || ChartUtils.formatColumnName(xCol)}: ${context[0].label}`,
                    label: (ctx) => `${ctx.dataset.label}: ${ChartUtils.formatFinancialValue(ctx.parsed.y, yCol)}`
                }
            },
            datalabels: { display: false } // Too cluttered for time series
        },
        scales: {
            x: {
                grid: { display: true, color: 'rgba(226, 232, 240, 0.5)' },
                ticks: { 
                    color: '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' }
                },
                title: {
                    display: true,
                    text: friendlyColumns[xCol] || ChartUtils.formatColumnName(xCol),
                    color: '#64748b',
                    font: { size: 12, family: 'Inter', weight: '600' }
                }
            },
            y: {
                grid: { color: 'rgba(226, 232, 240, 0.5)' },
                ticks: {
                    color: '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' },
                    callback: (value) => ChartUtils.formatFinancialValue(value, yCol, true)
                },
                title: {
                    display: true,
                    text: friendlyColumns[yCol] || ChartUtils.formatColumnName(yCol),
                    color: '#64748b',
                    font: { size: 12, family: 'Inter', weight: '600' }
                }
            }
        },
        elements: {
            point: { radius: 4, hoverRadius: 6, borderWidth: 2 },
            line: { borderWidth: 3, tension: 0.4 }
        },
        layout: { padding: { top: 20, right: 20, bottom: 10, left: 20 } }
    };
}

// Generate highly distinct color palette optimized for charts
// REMOVED: This function is now available in utils.js as ChartUtils.generateColorPalette

function createBarChartOptions(xCol, yCol, friendlyColumns, chartData) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
            legend: {
                display: false // Single dataset for company comparison
            },
            tooltip: {
                backgroundColor: $('html').hasClass('dark') ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                titleColor: $('html').hasClass('dark') ? '#f1f5f9' : '#0f172a',
                bodyColor: $('html').hasClass('dark') ? '#e2e8f0' : '#334155',
                borderColor: $('html').hasClass('dark') ? '#334155' : '#e2e8f0',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { size: 13, weight: 'bold', family: 'Inter' },
                bodyFont: { size: 12, family: 'Inter' },
                callbacks: {
                    title: (context) => {
                        const index = context[0].dataIndex;
                        return chartData[index]._company || chartData[index][xCol] || 'Unknown Company';
                    },
                    label: (ctx) => `${friendlyColumns[yCol] || ChartUtils.formatColumnName(yCol)}: ${ChartUtils.formatFinancialValue(ctx.parsed.y, yCol)}`
                }
            },
            datalabels: {
                display: chartData.length <= 8,
                color: $('html').hasClass('dark') ? '#f1f5f9' : '#1e293b',
                font: { size: 11, weight: 'bold', family: 'Inter' },
                formatter: (value) => ChartUtils.formatFinancialValue(value, yCol, true),
                anchor: 'end',
                align: 'top',
                offset: 4
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { 
                    color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' },
                    maxRotation: 45,
                    minRotation: 20,
                    autoSkip: false,
                    callback: function(value, index, values) {
                        // Show full label if short, else truncate and add ...
                        const label = this.getLabelForValue(value);
                        if (label && label.length > 16) {
                            return label.substring(0, 14) + '…';
                        }
                        return label;
                    }
                }
            },
            y: {
                grid: { color: $('html').hasClass('dark') ? 'rgba(51, 65, 85, 0.3)' : 'rgba(226, 232, 240, 0.5)' },
                ticks: {
                    color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                    font: { size: 11, family: 'Inter', weight: '500' },
                    callback: (value) => ChartUtils.formatFinancialValue(value, yCol, true)
                }
            }
        },
        elements: {
            bar: { 
                borderRadius: 6, 
                borderSkipped: false, 
                borderWidth: 2,
                // Enhanced hover effects
                hoverBorderWidth: 3,
                hoverBorderRadius: 8
            }
        },
        layout: { padding: { top: 30, right: 20, bottom: 10, left: 20 } }
    };
}

// Enhanced chart selector population with LINE CHART = YEAR PRIORITY
function populateChartSelectors(dataContext) {
    const numericColumns = ChartUtils.getColumnsByType('numeric', dataContext);
    const categoricalColumns = ChartUtils.getColumnsByType('categorical', dataContext);
    const chartType = $('#chartModal .chart-type-toggle button.active').data('type');


    const populateSelect = (selector, columns, preferred) => {
        const select = $(selector).empty();
        if (columns.length === 0) {
            select.append('<option value="">No data available</option>');
        } else {
            let friendlyColumns = {};
            
            // Get friendly columns from appropriate source
            if (dataContext.currentMultiSheetData && dataContext.chartSheetSelectorValue === 'combined') {
                // For combined view, use friendly columns from first successful sheet
                const firstSheet = dataContext.currentSheets.find(sheet => sheet.success !== false && sheet.data && sheet.data.length > 0);
                friendlyColumns = firstSheet?.friendly_columns || {};
            } else if (dataContext.currentMultiSheetData && dataContext.chartSheetSelectorValue !== '') {
                const sheetIndex = parseInt(dataContext.chartSheetSelectorValue);
                friendlyColumns = dataContext.currentSheets[sheetIndex]?.friendly_columns || {};
            } else if (dataContext.lastApiResponse) {
                friendlyColumns = dataContext.lastApiResponse.friendly_columns || {};
            }
            
            columns.forEach(col => {
                const friendlyName = friendlyColumns[col] || ChartUtils.formatColumnName(col);
                select.append(`<option value="${col}">${friendlyName}</option>`);
            });
            
            if (preferred) {
                let pref = preferred.find(p => columns.includes(p));
                if (pref) select.val(pref);
            }
        }
    };
    
    if (chartType === 'line') {
        // 🚨 LINE CHARTS: ONLY year/time columns allowed for X-axis
        
        // Year columns - ONLY these are allowed for line charts
        const yearColumns = categoricalColumns.filter(col => {
            const colLower = col.toLowerCase();
            return colLower.includes('year') || 
                   colLower.includes('calendaryear') ||
                   colLower === 'yr' ||
                   colLower === 'fiscal_year' ||
                   colLower.includes('date') ||
                   colLower.includes('period') ||
                   colLower.includes('time');
        });
        
        
        // X-axis: ONLY year columns allowed for line charts
        populateSelect('#chartXAxis', yearColumns, 
                      ['year', 'calendarYear', 'calendaryear', 'fiscal_year', 'fiscalYear', 'yr', 'date', 'period', 'time']);
        
        // 🎯 FORCE year selection - always select the first available year column
        if (yearColumns.length > 0 && !$('#chartXAxis').val()) {
            $('#chartXAxis').val(yearColumns[0]);
        }
        
        // Y-axis: All numeric columns available
        populateSelect('#chartYAxis', numericColumns, 
                      ['revenue', 'segment_revenue', 'netIncome', 'totalRevenue', 'marketCap']);
    } else {
        // For BAR CHARTS: Use all categorical columns for X-axis
        
        // 🎯 For bar charts, also prioritize timeframe columns first if available
        const timeframeColsBar = categoricalColumns.filter(col => {
            const colLower = col.toLowerCase();
            return colLower.includes('year') || colLower.includes('date') || colLower.includes('period') || colLower.includes('time');
        });
        const nonTimeframeCols = categoricalColumns.filter(col => {
            const colLower = col.toLowerCase();
            return !colLower.includes('year') && !colLower.includes('date') && !colLower.includes('period') && !colLower.includes('time');
        });
        const barXAxisOptions = [...timeframeColsBar, ...nonTimeframeCols];
        
        populateSelect('#chartXAxis', barXAxisOptions, 
                      ['year', 'calendarYear', 'calendaryear', 'fiscal_year', 'fiscalYear', 'yr', 'date', 'companyName', 'company_name', 'symbol']);
        
        // 🎯 FORCE timeframe selection if available for bar charts too
        if (timeframeColsBar.length > 0 && !$('#chartXAxis').val()) {
            $('#chartXAxis').val(timeframeColsBar[0]);
        }
        populateSelect('#chartYAxis', numericColumns, 
                      ['mktCap', 'revenue', 'segment_revenue', 'netIncome', 'marketCap']);
    }
}

// Export functions for use in other files
window.ChartFunctions = {
    createStandardChart,
    createTimeSeriesChartOptions,
    createBarChartOptions,
    createStandardChartOptions,
    generateColorPalette,
    populateChartSelectors,
    // New moved functions
    createTimeSeriesChart,
    createSegmentTimeSeriesChart,
    createMultiCompanyBarChart,
    prepareCombinedChartData,
    isMultiSheetTimeSeriesData,
    destroyExistingChart
};

// ===== MOVED FROM INDEX.HTML =====
// Self-contained chart creation functions

// Enhanced time series chart creation with proper multi-company support
function createTimeSeriesChart(dataSource, xCol, yCol, friendlyColumns) {
    const dataPointsToShow = $('#chartDataPointsSelect').val();
    
    // Group data by company
    const companiesData = {};
    dataSource.forEach(row => {
        const company = row._company || row.company || 'Unknown';
        if (!companiesData[company]) {
            companiesData[company] = [];
        }
        companiesData[company].push(row);
    });
    
    
    // Sort each company's data by time
    Object.keys(companiesData).forEach(company => {
        companiesData[company].sort((a, b) => {
            const aTime = parseFloat(a[xCol]) || 0;
            const bTime = parseFloat(b[xCol]) || 0;
            return aTime - bTime;
        });
    });
    
    // Get all unique time points and sort them
    let allTimePoints = [...new Set(dataSource.map(row => row[xCol]))].sort((a, b) => {
        const aNum = parseFloat(a) || 0;
        const bNum = parseFloat(b) || 0;
        return aNum - bNum;
    });
    
    // Limit time points if not showing all data (time period filtering)
    if (dataPointsToShow !== 'all') {
        const yearsToShow = parseInt(dataPointsToShow) || 10;
        const originalLength = allTimePoints.length;
        // For time series, take the most recent time points
        allTimePoints = allTimePoints.slice(-yearsToShow);
    }
    
    
    const colors = ChartUtils.generateColorPalette(Object.keys(companiesData).length);
    const datasets = Object.entries(companiesData).map(([company, data], index) => {
        // Create data points for each time point (fill missing with null)
        const chartData = allTimePoints.map(timePoint => {
            const dataPoint = data.find(row => row[xCol] === timePoint);
            return dataPoint ? ChartUtils.parseFinancialNumber(dataPoint[yCol]) : null;
        });
        
        
        return {
            label: company,
            data: chartData,
            borderColor: colors[index],
            backgroundColor: colors[index] + '20',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: colors[index],
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            spanGaps: false // Don't connect across missing data
        };
    }).filter(dataset => {
        // Only include datasets that have at least one non-null data point in the selected timeframe
        return dataset.data.some(value => value !== null && value !== undefined);
    });
    
    return {
        data: { labels: allTimePoints, datasets },
        options: createTimeSeriesChartOptions(xCol, yCol, friendlyColumns)
    };
}

function createSegmentTimeSeriesChart(dataSource, xCol, yCol, friendlyColumns, seriesInfo) {
    const dataPointsToShow = $('#chartDataPointsSelect').val();
    
    // Group data by series (segments/categories)
    const seriesData = {};
    const seriesColumn = seriesInfo.seriesColumn;
    
    dataSource.forEach(row => {
        const seriesName = row[seriesColumn] || 'Unknown';
        if (!seriesData[seriesName]) {
            seriesData[seriesName] = [];
        }
        seriesData[seriesName].push(row);
    });
    
    
    // Sort each series data by time
    Object.keys(seriesData).forEach(seriesName => {
        seriesData[seriesName].sort((a, b) => {
            const aTime = parseFloat(a[xCol]) || 0;
            const bTime = parseFloat(b[xCol]) || 0;
            return aTime - bTime;
        });
    });
    
    // Get all unique time points and sort them
    let allTimePoints = [...new Set(dataSource.map(row => row[xCol]))].sort((a, b) => {
        const aNum = parseFloat(a) || 0;
        const bNum = parseFloat(b) || 0;
        return aNum - bNum;
    });
    
    // Limit time points if not showing all data (time period filtering)
    if (dataPointsToShow !== 'all') {
        const yearsToShow = parseInt(dataPointsToShow) || 10;
        const originalLength = allTimePoints.length;
        // For time series, take the most recent time points
        allTimePoints = allTimePoints.slice(-yearsToShow);
    }
    
    
    const colors = ChartUtils.generateColorPalette(Object.keys(seriesData).length);
    const datasets = Object.entries(seriesData).map(([seriesName, data], index) => {
        // Create data points for each time point (fill missing with null)
        const chartData = allTimePoints.map(timePoint => {
            const dataPoint = data.find(row => row[xCol] === timePoint);
            return dataPoint ? ChartUtils.parseFinancialNumber(dataPoint[yCol]) : null;
        });
        
        
        return {
            label: seriesName,
            data: chartData,
            borderColor: colors[index],
            backgroundColor: colors[index] + '20',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: colors[index],
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            spanGaps: false
        };
    }).filter(dataset => {
        // Only include datasets that have at least one non-null data point in the selected timeframe
        return dataset.data.some(value => value !== null && value !== undefined);
    });
    
    return {
        data: { labels: allTimePoints, datasets },
        options: createTimeSeriesChartOptions(xCol, yCol, friendlyColumns)
    };
}

// Create multi-company bar chart
function createMultiCompanyBarChart(dataSource, xCol, yCol, friendlyColumns) {
    const dataPointsToShow = $('#chartDataPointsSelect').val();
    
    // If we have year data, we might want to filter for a specific year
    let filteredData = dataSource;
    
    // Check if we should filter by year for cleaner comparison
    if (dataSource.some(row => row.year || row.YEAR || row.Year)) {
        const years = [...new Set(dataSource.map(row => row.year || row.YEAR || row.Year))].filter(Boolean);
        if (years.length > 1) {
            // Use the most recent year
            const latestYear = Math.max(...years.map(y => parseInt(y) || 0));
            filteredData = dataSource.filter(row => {
                const rowYear = row.year || row.YEAR || row.Year;
                return parseInt(rowYear) === latestYear;
            });
        }
    }
    
    // Sort by the metric we're charting and take top N
    filteredData.sort((a, b) => {
        const aVal = ChartUtils.parseFinancialNumber(a[yCol]) || 0;
        const bVal = ChartUtils.parseFinancialNumber(b[yCol]) || 0;
        return bVal - aVal; // Descending order
    });
    
    const chartData = (dataPointsToShow === 'all') ? filteredData : filteredData.slice(0, parseInt(dataPointsToShow) || 10);
    const colors = ChartUtils.generateColorPalette(chartData.length);
    
    const labels = chartData.map(row => {
        // 🏢 PRIORITIZE company name for multi-company data
        const companyName = row._company || row.company || row.companyName || row.company_name;
        const xValue = row[xCol];
        
        // Use company name if available, otherwise use X column value
        let label = companyName || xValue || 'Unknown';
        
        // Clean up company names but don't truncate too aggressively
        if (label && label.length > 20) {
            let cleanLabel = label
                .replace(/ Inc\.?$/, '')
                .replace(/ Corp\.?$/, '')
                .replace(/ Ltd\.?$/, '')
                .replace(/ LLC\.?$/, '')
                .replace(/ Co\.?$/, '');
            
            if (cleanLabel.length > 20) {
                return cleanLabel.substring(0, 18) + '..';
            }
            return cleanLabel;
        }
        return label;
    });
    
    
    return {
        data: {
            labels: labels,
            datasets: [{
                label: friendlyColumns[yCol] || ChartUtils.formatColumnName(yCol),
                data: chartData.map(row => ChartUtils.parseFinancialNumber(row[yCol])),
                backgroundColor: colors.map(c => c + '80'), // Semi-transparent fill
                borderColor: colors, // Solid border for each company
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false,
                // Add hover effects
                hoverBackgroundColor: colors.map(c => c + 'A0'), // More opaque on hover
                hoverBorderColor: colors.map(c => c + 'FF'), // Solid on hover
                hoverBorderWidth: 3
            }]
        },
        options: createBarChartOptions(xCol, yCol, friendlyColumns, chartData)
    };
}

// Enhanced combined chart data preparation with proper time series support
function prepareCombinedChartData(currentSheets) {
    const successfulSheets = currentSheets.filter(sheet => sheet.success !== false && sheet.data && sheet.data.length > 0);
    
    if (successfulSheets.length === 0) {
        return { success: false, error: 'No successful sheets with data available.' };
    }
    
    // Find common columns across all sheets
    const baseColumns = successfulSheets[0].columns || [];
    const commonColumns = baseColumns.filter(col => 
        successfulSheets.every(sheet => sheet.columns && sheet.columns.includes(col))
    );
    
    if (commonColumns.length === 0) {
        return { success: false, error: 'No common columns found across sheets.' };
    }
    
    
    // Combine data from all sheets, adding company identifier
    const combinedData = [];
    successfulSheets.forEach((sheet, index) => {
        sheet.data.forEach(row => {
            const newRow = { ...row };
            // Add company identifier for multi-company charts
            newRow._company = sheet.company || `Company ${index + 1}`;
            newRow._sheet_name = sheet.sheet_name;
            combinedData.push(newRow);
        });
    });
    
    // Use friendly columns from first sheet
    const friendlyColumns = successfulSheets[0].friendly_columns || {};
    
    
    return {
        success: true,
        data: combinedData,
        columns: commonColumns,
        friendlyColumns: friendlyColumns,
        title: `Combined Analysis (${successfulSheets.length} companies)`
    };
}

// Helper function to detect if we have time series data across multiple sheets
function isMultiSheetTimeSeriesData(currentMultiSheetData, currentSheets, chartSheetSelectorValue) {
    if (!currentMultiSheetData || chartSheetSelectorValue !== 'combined') {
        return false;
    }
    
    const successfulSheets = currentSheets.filter(sheet => sheet.success !== false && sheet.data && sheet.data.length > 0);
    if (successfulSheets.length < 2) return false;
    
    // Check if sheets have year/date columns and represent different companies or time periods
    const hasYearColumn = successfulSheets.some(sheet => 
        sheet.columns && sheet.columns.some(col => {
            const colLower = col.toLowerCase();
            return colLower.includes('year') || colLower.includes('calendaryear') || colLower.includes('date');
        })
    );
    
    // Check if we have multiple companies (different sheet companies)
    const uniqueCompanies = new Set(successfulSheets.map(sheet => sheet.company));
    const hasMultipleCompanies = uniqueCompanies.size > 1;
    
    
    return hasYearColumn && hasMultipleCompanies;
}

// Properly destroy chart to prevent conflicts
function destroyExistingChart(chartInstance) {
    if (chartInstance) {
        try {
            chartInstance.destroy();
        } catch (e) {
        }
        chartInstance = null;
    }
    
    // Clear canvas context
    const canvas = document.getElementById('interactiveChart');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Hide any error messages
    $('#chartError').addClass('hidden');
    
    return null; // Return null to reset chartInstance
}
