"""Alerting system for monitoring."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: str
    sku: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """
    Manages alerts from various monitoring components.
    """
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self._alerts: List[Alert] = []
        self._alert_counter = 0
    
    def create_alert(
        self,
        alert_type: str,
        sku: str,
        severity: str,
        message: str,
        metadata: Dict = None,
    ) -> Alert:
        """Create and store a new alert."""
        self._alert_counter += 1
        
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            type=alert_type,
            sku=sku,
            severity=AlertSeverity(severity) if isinstance(severity, str) else severity,
            message=message,
            metadata=metadata or {},
        )
        
        self._alerts.append(alert)
        
        # Trim old alerts
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[-self.max_alerts:]
        
        return alert
    
    def get_alerts(
        self,
        sku: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = self._alerts
        
        if sku:
            alerts = [a for a in alerts if a.sku == sku]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts[-limit:]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        unacked = [a for a in self._alerts if not a.acknowledged]
        
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in unacked if a.severity == severity)
            by_severity[severity.value] = count
        
        by_type = {}
        for alert in unacked:
            by_type[alert.type] = by_type.get(alert.type, 0) + 1
        
        return {
            'total_alerts': len(self._alerts),
            'unacknowledged': len(unacked),
            'by_severity': by_severity,
            'by_type': by_type,
            'latest_alert': self._alerts[-1].timestamp.isoformat() if self._alerts else None,
        }


# Singleton instance
alert_manager = AlertManager()
