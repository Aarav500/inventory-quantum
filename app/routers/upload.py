"""CSV upload router."""

import uuid
from datetime import datetime
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.s3 import s3_service
from app.services.validation import data_validator, ValidationError
from app.models.responses import APIResponse, UploadResponse

router = APIRouter()

# In-memory storage for when S3 is not available
_local_data_store: dict = {}


@router.post("/", response_model=APIResponse[UploadResponse])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with inventory data.
    
    Expected columns:
    - date: Date of the record (required)
    - sku: Stock Keeping Unit identifier (required)  
    - quantity_sold: Units sold (required)
    - quantity_on_hand: Current inventory level (optional)
    - price: Unit price (optional)
    - lead_time_days: Lead time in days (optional)
    - holding_cost: Holding cost per unit per day (optional)
    - ordering_cost: Fixed ordering cost (optional)
    - stockout_cost: Stockout cost per unit (optional)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file content
    content = await file.read()
    
    # Validate and parse
    try:
        df, warnings = data_validator.validate_csv(content)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get dataset stats
    stats = data_validator.get_dataset_stats(df)
    
    # Generate S3 key
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"uploads/{timestamp}_{file.filename}"
    
    # Try to upload to S3
    if s3_service.is_available():
        success = s3_service.upload_file(
            BytesIO(content),
            s3_key,
            content_type="text/csv"
        )
        if not success:
            warnings.append("Failed to upload to S3, stored locally")
            s3_key = "local"
            _local_data_store[file.filename] = df
    else:
        warnings.append("S3 not available, stored locally")
        s3_key = "local"
        _local_data_store[file.filename] = df
    
    # Also store processed DataFrame for quick access
    _local_data_store['latest'] = df
    _local_data_store['latest_stats'] = stats
    
    return APIResponse(
        success=True,
        data=UploadResponse(
            filename=file.filename,
            s3_key=s3_key,
            records_processed=stats['total_records'],
            skus_found=stats['skus'],
            date_range=(stats['date_range'][0], stats['date_range'][1]),
            validation_warnings=warnings,
        ),
        message=f"Successfully processed {stats['total_records']} records"
    )


@router.get("/data")
async def get_uploaded_data():
    """Get the latest uploaded data summary."""
    if 'latest' not in _local_data_store:
        raise HTTPException(status_code=404, detail="No data uploaded yet")
    
    stats = _local_data_store.get('latest_stats', {})
    return APIResponse(
        success=True,
        data=stats,
        message="Data available"
    )


@router.get("/skus")
async def list_skus():
    """List all available SKUs in the uploaded data."""
    if 'latest' not in _local_data_store:
        raise HTTPException(status_code=404, detail="No data uploaded yet")
    
    df = _local_data_store['latest']
    skus = df['sku'].unique().tolist()
    
    return APIResponse(
        success=True,
        data={"skus": skus, "count": len(skus)},
        message=f"Found {len(skus)} SKUs"
    )


def get_data():
    """Get the latest uploaded DataFrame (for internal use)."""
    return _local_data_store.get('latest')
