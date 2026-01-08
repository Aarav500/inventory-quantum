// Feature explanations data
const FEATURE_EXPLANATIONS = {
    date: {
        name: 'Date',
        icon: '📅',
        required: true,
        description: 'Timestamps enable temporal pattern recognition including seasonality, trends, and day-of-week effects.',
        contribution: 0.15,
        color: '#3b82f6',
        whyAccuracy: 'Captures weekly patterns (weekends vs weekdays) and monthly seasonality.'
    },
    sku: {
        name: 'SKU',
        icon: '🏷️',
        required: true,
        description: 'Product identifier allows independent forecasting per item with unique demand patterns.',
        contribution: 0.10,
        color: '#3b82f6',
        whyAccuracy: 'Each product has unique behavior - aggregating loses critical information.'
    },
    quantity_sold: {
        name: 'Quantity Sold',
        icon: '📈',
        required: true,
        description: 'Historical sales data is the primary target variable for demand forecasting.',
        contribution: 0.25,
        color: '#3b82f6',
        whyAccuracy: 'The core signal - lag features (lag_1, lag_7) capture autocorrelation.'
    },
    quantity_on_hand: {
        name: 'Inventory Level',
        icon: '📦',
        required: false,
        description: 'Current stock levels help detect stockout-induced demand suppression.',
        contribution: 0.12,
        color: '#10b981',
        whyAccuracy: 'Zero inventory may suppress observed demand - model adjusts for this bias.'
    },
    price: {
        name: 'Price',
        icon: '💵',
        required: false,
        description: 'Price elasticity affects demand - promotions and discounts create demand spikes.',
        contribution: 0.10,
        color: '#f59e0b',
        whyAccuracy: 'Price drops correlate with demand increases via elasticity models.'
    },
    lead_time_days: {
        name: 'Lead Time',
        icon: '⏱️',
        required: false,
        description: 'Supplier lead time affects safety stock calculations and reorder points.',
        contribution: 0.05,
        color: '#f59e0b',
        whyAccuracy: 'Critical for optimization, less for pure forecasting accuracy.'
    },
    holding_cost: {
        name: 'Holding Cost',
        icon: '🏠',
        required: false,
        description: 'Cost per unit per period for storage - optimizes inventory levels.',
        contribution: 0.03,
        color: '#ef4444',
        whyAccuracy: 'Used in QUBO cost function, not in forecast model directly.'
    },
    ordering_cost: {
        name: 'Ordering Cost',
        icon: '📋',
        required: false,
        description: 'Fixed cost per order - affects Economic Order Quantity calculations.',
        contribution: 0.03,
        color: '#ef4444',
        whyAccuracy: 'Drives EOQ trade-off, not forecast accuracy.'
    },
    stockout_cost: {
        name: 'Stockout Cost',
        icon: '⚠️',
        required: false,
        description: 'Penalty for unmet demand - determines service level targets.',
        contribution: 0.02,
        color: '#ef4444',
        whyAccuracy: 'Affects risk-aware optimization (CVaR), not pure forecast.'
    },
    promotion: {
        name: 'Promotion Flag',
        icon: '🎉',
        required: false,
        description: 'Binary indicator for promotional periods - captures demand spikes.',
        contribution: 0.08,
        color: '#8b5cf6',
        whyAccuracy: 'Promotions can 2-3x demand - critical for accurate peak forecasting.'
    },
    category: {
        name: 'Category',
        icon: '🗂️',
        required: false,
        description: 'Product category enables cross-learning between similar items.',
        contribution: 0.04,
        color: '#8b5cf6',
        whyAccuracy: 'Hierarchical models share patterns across category (e.g., seasonal toys).'
    },
    region: {
        name: 'Region',
        icon: '🌍',
        required: false,
        description: 'Geographic segmentation captures regional demand differences.',
        contribution: 0.03,
        color: '#8b5cf6',
        whyAccuracy: 'Regional events and weather affect local demand patterns.'
    }
};

// Sample datasets metadata - FIX: Use absolute static paths
const SAMPLE_DATASETS = {
    1: { name: 'Minimal', file: '/static/sample_data/1_minimal.csv', columns: 3, expectedAccuracy: 70 },
    2: { name: 'Basic', file: '/static/sample_data/2_basic.csv', columns: 4, expectedAccuracy: 78 },
    3: { name: 'Standard', file: '/static/sample_data/3_standard.csv', columns: 6, expectedAccuracy: 85 },
    4: { name: 'Advanced', file: '/static/sample_data/4_advanced.csv', columns: 8, expectedAccuracy: 90 },
    5: { name: 'Complete', file: '/static/sample_data/5_complete.csv', columns: 12, expectedAccuracy: 95 }
};

// Global state
let currentData = null;
let currentDataName = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initUploadArea();
    initFeatureExplanations();
    loadStoredData();
});

// Theme Toggle
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const btn = document.querySelector('.theme-toggle');
    if (btn) {
        btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
        btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
    }
}

function initUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (!uploadArea || !fileInput) return;

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });
}

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    // 1. Process Validated CSV Client-Side for Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        const data = parseCSV(content);
        currentData = data;
        currentDataName = file.name;
        displayDataPreview(data);
        assessDataQuality(data);
        saveDataToStorage(data, file.name);
    };
    reader.readAsText(file);

    // 2. Upload to Server for Backend Processing
    await uploadToServer(file);
}

async function uploadToServer(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.error('Upload failed:', await response.text());
        } else {
            console.log('File uploaded to server successfully');
        }
    } catch (e) {
        console.error('Error uploading file:', e);
    }
}

function parseCSV(content) {
    const lines = content.trim().split('\n');
    if (lines.length < 2) return { headers: [], rows: [] };

    const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/\s+/g, '_'));
    const rows = lines.slice(1).map(line => {
        const values = line.split(',');
        const row = {};
        headers.forEach((h, i) => row[h] = values[i]?.trim() || '');
        return row;
    });

    return { headers, rows };
}

function dataToCSV(data) {
    if (!data || !data.headers || !data.rows) return '';
    const headers = data.headers.join(',');
    const rows = data.rows.map(row => data.headers.map(h => row[h]).join(',')).join('\n');
    return headers + '\n' + rows;
}

async function loadSample(level) {
    const sample = SAMPLE_DATASETS[level];
    try {
        const response = await fetch(sample.file);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const content = await response.text();
        const data = parseCSV(content);
        currentData = data;
        currentDataName = sample.name;
        displayDataPreview(data);
        assessDataQuality(data);
        saveDataToStorage(data, sample.name);

        // Convert to File object and upload to server
        const file = new File([content], `${sample.name.toLowerCase()}_sample.csv`, { type: 'text/csv' });
        await uploadToServer(file);

    } catch (e) {
        console.error('Error loading sample:', e);
        // Generate demo data if file not found
        generateDemoData(level);
    }
}

async function generateDemoData(level) {
    const baseColumns = ['date', 'sku', 'quantity_sold'];
    const additionalColumns = {
        2: ['quantity_on_hand'],
        3: ['quantity_on_hand', 'price', 'lead_time_days'],
        4: ['quantity_on_hand', 'price', 'lead_time_days', 'holding_cost', 'ordering_cost'],
        5: ['quantity_on_hand', 'price', 'lead_time_days', 'holding_cost', 'ordering_cost', 'stockout_cost', 'category', 'region', 'promotion']
    };

    const headers = [...baseColumns, ...(additionalColumns[level] || [])];
    const rows = [];

    for (let d = 0; d < 30; d++) {
        const date = new Date(2024, 0, d + 1);
        const row = {
            date: date.toISOString().split('T')[0],
            sku: 'PROD-A',
            quantity_sold: Math.floor(40 + Math.random() * 30)
        };

        if (headers.includes('quantity_on_hand')) row.quantity_on_hand = Math.floor(50 + Math.random() * 150);
        if (headers.includes('price')) row.price = '29.99';
        if (headers.includes('lead_time_days')) row.lead_time_days = '7';
        if (headers.includes('holding_cost')) row.holding_cost = '0.10';
        if (headers.includes('ordering_cost')) row.ordering_cost = '50.0';
        if (headers.includes('stockout_cost')) row.stockout_cost = '10.0';
        if (headers.includes('category')) row.category = 'Electronics';
        if (headers.includes('region')) row.region = 'North';
        if (headers.includes('promotion')) row.promotion = Math.random() > 0.85 ? '1' : '0';

        rows.push(row);
    }

    currentData = { headers, rows };
    currentDataName = SAMPLE_DATASETS[level].name;
    displayDataPreview(currentData);
    assessDataQuality(currentData);
    saveDataToStorage(currentData, currentDataName);

    // Upload generated demo data to server as well
    const csvContent = dataToCSV(currentData);
    const file = new File([csvContent], `demo_${currentDataName.toLowerCase()}.csv`, { type: 'text/csv' });
    await uploadToServer(file);
}

function displayDataPreview(data) {
    const preview = document.getElementById('data-preview');
    const stats = document.getElementById('preview-stats');
    const table = document.getElementById('preview-table');

    if (!preview) return;
    preview.style.display = 'block';

    // Stats
    const uniqueSkus = new Set(data.rows.map(r => r.sku)).size;
    const dateRange = data.rows.length > 0 ?
        `${data.rows[0].date} to ${data.rows[data.rows.length - 1].date}` : 'N/A';

    stats.innerHTML = `
        <div class="stat-item"><div class="stat-value">${data.rows.length}</div><div class="stat-label">Records</div></div>
        <div class="stat-item"><div class="stat-value">${data.headers.length}</div><div class="stat-label">Columns</div></div>
        <div class="stat-item"><div class="stat-value">${uniqueSkus}</div><div class="stat-label">SKUs</div></div>
    `;

    // Table
    const headerRow = `<tr>${data.headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
    const bodyRows = data.rows.slice(0, 10).map(row =>
        `<tr>${data.headers.map(h => `<td>${row[h] || '-'}</td>`).join('')}</tr>`
    ).join('');

    table.innerHTML = `<thead>${headerRow}</thead><tbody>${bodyRows}</tbody>`;

    preview.scrollIntoView({ behavior: 'smooth' });
}

function assessDataQuality(data) {
    const qualityDiv = document.getElementById('data-quality');
    const indicators = document.getElementById('quality-indicators');

    if (!qualityDiv) return;
    qualityDiv.style.display = 'block';

    const requiredCols = ['date', 'sku', 'quantity_sold'];
    const optionalCols = ['quantity_on_hand', 'price', 'lead_time_days', 'holding_cost', 'ordering_cost', 'stockout_cost', 'promotion', 'category', 'region'];

    let html = '';

    requiredCols.forEach(col => {
        const present = data.headers.includes(col);
        const icon = present ? '✓' : '✗';
        const status = present ? 'quality-good' : 'quality-missing';
        html += `<div class="quality-item"><div class="quality-status ${status}">${icon}</div><div>${col}</div></div>`;
    });

    optionalCols.forEach(col => {
        const present = data.headers.includes(col);
        const icon = present ? '✓' : '○';
        const status = present ? 'quality-good' : 'quality-warning';
        html += `<div class="quality-item"><div class="quality-status ${status}">${icon}</div><div>${col}</div></div>`;
    });

    indicators.innerHTML = html;
}

function initFeatureExplanations() {
    const container = document.getElementById('feature-explanations');
    if (!container) return;

    let html = '';
    Object.entries(FEATURE_EXPLANATIONS).forEach(([key, feature]) => {
        const reqBadge = feature.required ? '<span class="tag tag-required">Required</span>' : '<span class="tag" style="background:#6b7280;">Optional</span>';
        html += `
            <div class="feature-explain-card">
                <h4>${feature.icon} ${feature.name} ${reqBadge}</h4>
                <p>${feature.description}</p>
                <p><strong>Why it improves accuracy:</strong> ${feature.whyAccuracy}</p>
                <div class="contribution-bar">
                    <div class="contribution-fill" style="width: ${feature.contribution * 100}%; background: ${feature.color};"></div>
                </div>
                <div class="contribution-label">
                    <span>Contribution to Accuracy</span>
                    <span>${(feature.contribution * 100).toFixed(0)}%</span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function saveDataToStorage(data, name) {
    try {
        localStorage.setItem('inventoryData', JSON.stringify(data));
        localStorage.setItem('inventoryDataName', name);
    } catch (e) {
        console.warn('Could not save to localStorage:', e);
    }
}

function loadStoredData() {
    try {
        const stored = localStorage.getItem('inventoryData');
        const name = localStorage.getItem('inventoryDataName');
        if (stored) {
            currentData = JSON.parse(stored);
            currentDataName = name;
        }
    } catch (e) {
        console.warn('Could not load from localStorage:', e);
    }
}

function proceedToForecasting() {
    window.location.href = 'forecasting.html';
}

// Export for other pages
window.inventoryApp = {
    getData: () => currentData,
    getDataName: () => currentDataName,
    FEATURE_EXPLANATIONS,
    SAMPLE_DATASETS
};
