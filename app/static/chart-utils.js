/**
 * Chart.js Utility Functions for Inventory Quantum
 * Provides reusable chart configurations and helpers
 */

// Chart.js default configuration
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() || '#e0e0ff',
                font: { family: 'Inter, sans-serif' }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(26, 26, 62, 0.95)',
            titleColor: '#e0e0ff',
            bodyColor: '#8b8ba7',
            borderColor: 'rgba(124, 58, 237, 0.5)',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12
        }
    },
    scales: {
        x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#8b8ba7' }
        },
        y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#8b8ba7' }
        }
    }
};

// Gradient creators
function createGradient(ctx, colorStart, colorEnd) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

// Common color palettes
const chartColors = {
    primary: '#7c3aed',
    secondary: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#3b82f6',
    purple: '#8b5cf6',
    pink: '#ec4899'
};

// Line chart configuration
function createLineChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...chartDefaults,
            ...options,
            elements: {
                line: { tension: 0.4, borderWidth: 2 },
                point: { radius: 4, hoverRadius: 6 }
            }
        }
    });
}

// Bar chart configuration
function createBarChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            ...chartDefaults,
            ...options,
            elements: {
                bar: { borderRadius: 4 }
            }
        }
    });
}

// Doughnut chart configuration
function createDoughnutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: colors || Object.values(chartColors).slice(0, data.length),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right' }
            },
            cutout: '60%'
        }
    });
}

// Demand forecast chart with confidence intervals
function createForecastChart(canvasId, dates, actual, forecast, lowerBound, upperBound) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Actual',
                    data: actual,
                    borderColor: chartColors.primary,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'Forecast',
                    data: forecast,
                    borderColor: chartColors.success,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 3
                },
                {
                    label: 'Confidence Interval',
                    data: upperBound,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: '',
                    data: lowerBound,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    ...chartDefaults.plugins.legend,
                    labels: {
                        ...chartDefaults.plugins.legend.labels,
                        filter: (item) => item.text !== ''
                    }
                }
            }
        }
    });
}

// Anomaly visualization chart
function createAnomalyChart(canvasId, dates, values, anomalyIndices) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    const pointColors = values.map((_, i) =>
        anomalyIndices.includes(i) ? chartColors.danger : chartColors.primary
    );
    const pointRadius = values.map((_, i) =>
        anomalyIndices.includes(i) ? 8 : 4
    );

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Demand',
                data: values,
                borderColor: chartColors.primary,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointBackgroundColor: pointColors,
                pointRadius: pointRadius,
                pointHoverRadius: 8
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        afterLabel: (context) => {
                            if (anomalyIndices.includes(context.dataIndex)) {
                                return '⚠️ Anomaly Detected';
                            }
                            return '';
                        }
                    }
                }
            }
        }
    });
}

// Export functions
window.chartUtils = {
    createLineChart,
    createBarChart,
    createDoughnutChart,
    createForecastChart,
    createAnomalyChart,
    chartColors,
    chartDefaults
};
