"""Export router for CSV and Excel downloads."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from io import StringIO, BytesIO
import csv
from datetime import datetime

router = APIRouter()


def generate_demo_data():
    """Generate demo inventory data for export."""
    return [
        {"sku": "PROD-001", "name": "Widget A", "quantity": 150, "reorder_point": 50, "price": 29.99, "category": "Electronics"},
        {"sku": "PROD-002", "name": "Widget B", "quantity": 75, "reorder_point": 30, "price": 49.99, "category": "Electronics"},
        {"sku": "PROD-003", "name": "Gadget X", "quantity": 200, "reorder_point": 100, "price": 19.99, "category": "Accessories"},
        {"sku": "PROD-004", "name": "Gadget Y", "quantity": 45, "reorder_point": 25, "price": 39.99, "category": "Accessories"},
        {"sku": "PROD-005", "name": "Tool Pro", "quantity": 300, "reorder_point": 150, "price": 89.99, "category": "Tools"},
    ]


def generate_forecast_data():
    """Generate demo forecast data for export."""
    base_date = datetime.now()
    data = []
    for i in range(30):
        data.append({
            "date": (base_date.replace(day=1) if base_date.day + i > 28 else base_date).strftime("%Y-%m-%d"),
            "sku": "PROD-001",
            "predicted_demand": 50 + (i % 10) * 5,
            "lower_bound": 40 + (i % 10) * 4,
            "upper_bound": 60 + (i % 10) * 6,
            "confidence": 0.95
        })
    return data


@router.get("/csv/{data_type}")
async def export_csv(data_type: str):
    """Export data as CSV file.
    
    data_type: 'inventory', 'forecast', or 'report'
    """
    if data_type == "inventory":
        data = generate_demo_data()
    elif data_type == "forecast":
        data = generate_forecast_data()
    elif data_type == "report":
        data = generate_demo_data()  # Use inventory for report demo
    else:
        raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")
    
    if not data:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    
    # Reset stream position
    output.seek(0)
    
    filename = f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/excel/{data_type}")
async def export_excel(data_type: str):
    """Export data as Excel file (CSV with .xlsx extension for compatibility)."""
    # For simplicity, we export as CSV with Excel-compatible format
    # In production, you'd use openpyxl or xlsxwriter
    
    if data_type == "inventory":
        data = generate_demo_data()
    elif data_type == "forecast":
        data = generate_forecast_data()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")
    
    if not data:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    output.seek(0)
    
    filename = f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
