"""Integrations router for connecting to external business systems."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter()


class IntegrationProvider(str, Enum):
    SHOPIFY = "shopify"
    QUICKBOOKS = "quickbooks"
    SAP = "sap"
    WOOCOMMERCE = "woocommerce"
    MAGENTO = "magento"
    NETSUITE = "netsuite"
    SQUARE = "square"


class IntegrationStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    SYNCING = "syncing"
    PENDING = "pending"


class Integration(BaseModel):
    """An external system integration."""
    id: str
    provider: IntegrationProvider
    name: str
    status: IntegrationStatus
    connected_at: Optional[str] = None
    last_sync: Optional[str] = None
    sync_frequency: str = "hourly"
    items_synced: int = 0
    config: Dict = Field(default_factory=dict)
    error_message: Optional[str] = None


class IntegrationInfo(BaseModel):
    """Information about an available integration."""
    provider: IntegrationProvider
    name: str
    description: str
    features: List[str]
    requires_oauth: bool
    documentation_url: str
    logo_url: Optional[str] = None
    setup_complexity: str  # easy, medium, advanced


class ConnectRequest(BaseModel):
    """Request to connect an integration."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    store_url: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class SyncResult(BaseModel):
    """Result of a sync operation."""
    integration_id: str
    provider: str
    status: str
    items_synced: int
    items_created: int
    items_updated: int
    errors: int
    duration_seconds: float
    completed_at: str


# Available integrations info
available_integrations = [
    IntegrationInfo(
        provider=IntegrationProvider.SHOPIFY,
        name="Shopify",
        description="Sync inventory with your Shopify store. Real-time stock updates, order import, and product management.",
        features=[
            "Real-time inventory sync",
            "Order import & tracking",
            "Product catalog sync",
            "Multi-location support",
            "Webhook notifications"
        ],
        requires_oauth=True,
        documentation_url="https://shopify.dev/docs/api",
        setup_complexity="easy"
    ),
    IntegrationInfo(
        provider=IntegrationProvider.QUICKBOOKS,
        name="QuickBooks Online",
        description="Connect to QuickBooks for financial tracking, purchase orders, and vendor management.",
        features=[
            "Purchase order sync",
            "Vendor management",
            "Cost tracking",
            "Invoice generation",
            "Financial reporting"
        ],
        requires_oauth=True,
        documentation_url="https://developer.intuit.com",
        setup_complexity="medium"
    ),
    IntegrationInfo(
        provider=IntegrationProvider.SAP,
        name="SAP S/4HANA",
        description="Enterprise integration with SAP for complete supply chain management.",
        features=[
            "Material management",
            "Warehouse management",
            "Production planning",
            "Financial integration",
            "Advanced analytics"
        ],
        requires_oauth=True,
        documentation_url="https://api.sap.com",
        setup_complexity="advanced"
    ),
    IntegrationInfo(
        provider=IntegrationProvider.WOOCOMMERCE,
        name="WooCommerce",
        description="Sync with WordPress WooCommerce stores for inventory and order management.",
        features=[
            "Inventory sync",
            "Order import",
            "Product sync",
            "Stock level updates"
        ],
        requires_oauth=False,
        documentation_url="https://woocommerce.github.io/woocommerce-rest-api-docs/",
        setup_complexity="easy"
    ),
    IntegrationInfo(
        provider=IntegrationProvider.SQUARE,
        name="Square",
        description="Connect Square POS for retail inventory management.",
        features=[
            "POS inventory sync",
            "Transaction import",
            "Location management",
            "Real-time updates"
        ],
        requires_oauth=True,
        documentation_url="https://developer.squareup.com",
        setup_complexity="easy"
    ),
]

# Connected integrations (in-memory for demo)
connected_integrations: List[Integration] = [
    Integration(
        id="int-demo-001",
        provider=IntegrationProvider.SHOPIFY,
        name="Demo Shopify Store",
        status=IntegrationStatus.CONNECTED,
        connected_at=datetime.now().isoformat(),
        last_sync=datetime.now().isoformat(),
        sync_frequency="hourly",
        items_synced=1247,
        config={"store_url": "demo-store.myshopify.com"}
    )
]


@router.get("", response_model=List[Integration])
async def list_integrations():
    """
    List all configured integrations.
    
    Returns connected and pending integrations.
    """
    return connected_integrations


@router.get("/available", response_model=List[IntegrationInfo])
async def list_available_integrations():
    """
    List available integration providers.
    
    Shows all supported platforms that can be connected.
    """
    return available_integrations


@router.get("/available/{provider}", response_model=IntegrationInfo)
async def get_integration_info(provider: IntegrationProvider):
    """Get detailed information about an integration provider."""
    info = next((i for i in available_integrations if i.provider == provider), None)
    if not info:
        raise HTTPException(status_code=404, detail="Integration not found")
    return info


@router.post("/{provider}/connect", response_model=Integration)
async def connect_integration(provider: IntegrationProvider, request: ConnectRequest):
    """
    Connect to an external system.
    
    In production, this would initiate OAuth flow or validate API keys.
    Demo mode simulates a successful connection.
    """
    # Check if already connected
    existing = next((i for i in connected_integrations if i.provider == provider), None)
    if existing and existing.status == IntegrationStatus.CONNECTED:
        raise HTTPException(status_code=400, detail="Integration already connected")
    
    info = next((i for i in available_integrations if i.provider == provider), None)
    if not info:
        raise HTTPException(status_code=404, detail="Integration provider not supported")
    
    integration = Integration(
        id=f"int-{uuid.uuid4().hex[:8]}",
        provider=provider,
        name=f"{info.name} Integration",
        status=IntegrationStatus.CONNECTED,
        connected_at=datetime.now().isoformat(),
        last_sync=None,
        sync_frequency="hourly",
        items_synced=0,
        config={
            "store_url": request.store_url,
            "demo_mode": True
        }
    )
    
    connected_integrations.append(integration)
    
    return integration


@router.post("/{provider}/sync", response_model=SyncResult)
async def sync_integration(provider: IntegrationProvider):
    """
    Trigger a sync with the external system.
    
    Imports/exports data based on integration configuration.
    """
    integration = next((i for i in connected_integrations if i.provider == provider), None)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not connected")
    
    if integration.status != IntegrationStatus.CONNECTED:
        raise HTTPException(status_code=400, detail="Integration not in connected state")
    
    # Simulate sync
    import random
    items = random.randint(50, 500)
    
    integration.last_sync = datetime.now().isoformat()
    integration.items_synced += items
    
    return SyncResult(
        integration_id=integration.id,
        provider=provider.value,
        status="success",
        items_synced=items,
        items_created=random.randint(0, items // 5),
        items_updated=items - random.randint(0, items // 5),
        errors=0,
        duration_seconds=random.uniform(2.5, 15.0),
        completed_at=datetime.now().isoformat()
    )


@router.delete("/{provider}/disconnect")
async def disconnect_integration(provider: IntegrationProvider):
    """
    Disconnect an integration.
    
    Removes the connection but preserves synced data.
    """
    for i, integration in enumerate(connected_integrations):
        if integration.provider == provider:
            connected_integrations.pop(i)
            return {
                "message": f"{provider.value} integration disconnected",
                "data_preserved": True
            }
    
    raise HTTPException(status_code=404, detail="Integration not found")


@router.get("/{provider}/status")
async def get_integration_status(provider: IntegrationProvider):
    """Get current status of an integration."""
    integration = next((i for i in connected_integrations if i.provider == provider), None)
    
    if not integration:
        return {
            "provider": provider.value,
            "connected": False,
            "status": "not_configured"
        }
    
    return {
        "provider": provider.value,
        "connected": integration.status == IntegrationStatus.CONNECTED,
        "status": integration.status.value,
        "last_sync": integration.last_sync,
        "items_synced": integration.items_synced,
        "error": integration.error_message
    }


@router.get("/sync-history")
async def get_sync_history(
    provider: Optional[IntegrationProvider] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get sync history for integrations.
    
    Shows recent sync operations and their results.
    """
    # Demo sync history
    import random
    history = []
    
    for i in range(min(limit, 10)):
        prov = provider or random.choice(list(IntegrationProvider))
        history.append({
            "id": f"sync-{uuid.uuid4().hex[:8]}",
            "provider": prov.value,
            "status": random.choice(["success", "success", "success", "partial", "error"]),
            "items_synced": random.randint(10, 500),
            "errors": random.randint(0, 5),
            "timestamp": datetime.now().isoformat()
        })
    
    return history


@router.get("/stats")
async def get_integration_stats():
    """Get integration statistics summary."""
    connected = [i for i in connected_integrations if i.status == IntegrationStatus.CONNECTED]
    
    return {
        "total_available": len(available_integrations),
        "total_connected": len(connected),
        "total_items_synced": sum(i.items_synced for i in connected),
        "providers": {
            i.provider.value: {
                "connected": i.status == IntegrationStatus.CONNECTED,
                "items": i.items_synced
            }
            for i in connected_integrations
        },
        "supported_providers": [p.value for p in IntegrationProvider]
    }
