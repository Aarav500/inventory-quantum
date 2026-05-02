"""Alerts router for notification management."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter()


class AlertType(str, Enum):
    LOW_STOCK = "low_stock"
    STOCKOUT = "stockout"
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    FORECAST_DEVIATION = "forecast_deviation"
    REORDER_POINT = "reorder_point"


class NotificationChannel(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class AlertRule(BaseModel):
    """Alert rule configuration."""
    id: Optional[str] = None
    name: str
    alert_type: AlertType
    threshold: float = Field(..., description="Threshold value to trigger alert")
    sku_filter: Optional[str] = Field(None, description="SKU pattern to match (* for all)")
    channels: List[NotificationChannel] = [NotificationChannel.IN_APP]
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None


class Alert(BaseModel):
    """An alert instance."""
    id: str
    rule_id: str
    rule_name: str
    alert_type: AlertType
    sku: str
    message: str
    current_value: float
    threshold: float
    severity: str  # low, medium, high, critical
    triggered_at: str
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None


class TestNotificationRequest(BaseModel):
    """Request to send test notification."""
    channel: NotificationChannel
    email: Optional[str] = None
    webhook_url: Optional[str] = None


# In-memory storage for demo
alert_rules: List[AlertRule] = []
alerts: List[Alert] = []

# Initialize with demo rules
demo_rules = [
    AlertRule(
        id="rule-001",
        name="Critical Low Stock",
        alert_type=AlertType.LOW_STOCK,
        threshold=10,
        sku_filter="*",
        channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
    AlertRule(
        id="rule-002",
        name="Demand Spike Detection",
        alert_type=AlertType.DEMAND_SPIKE,
        threshold=50,  # 50% increase
        sku_filter="*",
        channels=[NotificationChannel.IN_APP],
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
    AlertRule(
        id="rule-003",
        name="Reorder Point Alert",
        alert_type=AlertType.REORDER_POINT,
        threshold=25,
        sku_filter="*",
        channels=[NotificationChannel.IN_APP, NotificationChannel.WEBHOOK],
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
]

# Demo alerts
demo_alerts = [
    Alert(
        id="alert-001",
        rule_id="rule-001",
        rule_name="Critical Low Stock",
        alert_type=AlertType.LOW_STOCK,
        sku="SKU-042",
        message="Stock level critically low: 8 units remaining",
        current_value=8,
        threshold=10,
        severity="critical",
        triggered_at=datetime.now().isoformat(),
        acknowledged=False
    ),
    Alert(
        id="alert-002",
        rule_id="rule-002",
        rule_name="Demand Spike Detection",
        alert_type=AlertType.DEMAND_SPIKE,
        sku="SKU-015",
        message="Demand increased by 65% in the last 24 hours",
        current_value=65,
        threshold=50,
        severity="high",
        triggered_at=datetime.now().isoformat(),
        acknowledged=False
    ),
    Alert(
        id="alert-003",
        rule_id="rule-003",
        rule_name="Reorder Point Alert",
        alert_type=AlertType.REORDER_POINT,
        sku="SKU-089",
        message="Stock at reorder point: 24 units",
        current_value=24,
        threshold=25,
        severity="medium",
        triggered_at=datetime.now().isoformat(),
        acknowledged=True,
        acknowledged_at=datetime.now().isoformat()
    ),
]

alert_rules.extend(demo_rules)
alerts.extend(demo_alerts)


@router.get("", response_model=List[Alert])
async def get_alerts(
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get all alerts.
    
    Returns list of triggered alerts, optionally filtered.
    """
    result = alerts.copy()
    
    if acknowledged is not None:
        result = [a for a in result if a.acknowledged == acknowledged]
    
    if severity:
        result = [a for a in result if a.severity == severity]
    
    return result[:limit]


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    for alert in alerts:
        if alert.id == alert_id:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.now().isoformat()
            return {"message": "Alert acknowledged", "alert": alert}
    
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/rules", response_model=List[AlertRule])
async def get_alert_rules():
    """Get all alert rules."""
    return alert_rules


@router.post("/rules", response_model=AlertRule)
async def create_alert_rule(rule: AlertRule):
    """
    Create a new alert rule.
    
    Rules define conditions that trigger alerts.
    """
    rule.id = f"rule-{uuid.uuid4().hex[:8]}"
    rule.created_at = datetime.now().isoformat()
    alert_rules.append(rule)
    return rule


@router.put("/rules/{rule_id}", response_model=AlertRule)
async def update_alert_rule(rule_id: str, updated_rule: AlertRule):
    """Update an existing alert rule."""
    for i, rule in enumerate(alert_rules):
        if rule.id == rule_id:
            updated_rule.id = rule_id
            updated_rule.created_at = rule.created_at
            alert_rules[i] = updated_rule
            return updated_rule
    
    raise HTTPException(status_code=404, detail="Rule not found")


@router.delete("/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    for i, rule in enumerate(alert_rules):
        if rule.id == rule_id:
            alert_rules.pop(i)
            return {"message": "Rule deleted", "id": rule_id}
    
    raise HTTPException(status_code=404, detail="Rule not found")


@router.post("/test")
async def test_notification(request: TestNotificationRequest):
    """
    Send a test notification.
    
    Useful for verifying notification channels are configured correctly.
    """
    if request.channel == NotificationChannel.EMAIL:
        if not request.email:
            raise HTTPException(status_code=400, detail="Email required for email channel")
        # In production, would send actual email
        return {
            "success": True,
            "message": f"Test email sent to {request.email}",
            "channel": "email",
            "note": "Demo mode - no actual email sent"
        }
    
    elif request.channel == NotificationChannel.WEBHOOK:
        if not request.webhook_url:
            raise HTTPException(status_code=400, detail="Webhook URL required")
        # In production, would POST to webhook
        return {
            "success": True,
            "message": f"Test webhook sent to {request.webhook_url}",
            "channel": "webhook",
            "note": "Demo mode - no actual webhook sent"
        }
    
    else:
        return {
            "success": True,
            "message": "Test in-app notification created",
            "channel": "in_app"
        }


@router.get("/history", response_model=List[Alert])
async def get_alert_history(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=500)
):
    """
    Get historical alerts.
    
    Returns past alerts for analysis and reporting.
    """
    # In production, would filter by date
    return alerts[:limit]


@router.get("/stats")
async def get_alert_stats():
    """Get alert statistics summary."""
    total = len(alerts)
    unacknowledged = len([a for a in alerts if not a.acknowledged])
    
    by_severity = {
        "critical": len([a for a in alerts if a.severity == "critical"]),
        "high": len([a for a in alerts if a.severity == "high"]),
        "medium": len([a for a in alerts if a.severity == "medium"]),
        "low": len([a for a in alerts if a.severity == "low"])
    }
    
    by_type = {}
    for alert_type in AlertType:
        by_type[alert_type.value] = len([a for a in alerts if a.alert_type == alert_type])
    
    return {
        "total_alerts": total,
        "unacknowledged": unacknowledged,
        "active_rules": len([r for r in alert_rules if r.is_active]),
        "by_severity": by_severity,
        "by_type": by_type
    }
