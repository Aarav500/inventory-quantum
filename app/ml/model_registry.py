"""
Model Registry for MLOps.

Tracks model versions, performance metrics, and enables rollback.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelVersion:
    """A versioned model."""
    model_id: str
    version: str
    model_type: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict
    status: str  # 'staging', 'production', 'archived'
    checksum: str


class ModelRegistry:
    """
    Simple model registry for tracking and managing model versions.
    """
    
    def __init__(self, storage_path: str = "./model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_path / "registry.json"
        self.models: Dict[str, List[ModelVersion]] = {}
        self._load()
    
    def _load(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.models[model_id] = [
                        ModelVersion(**v) for v in versions
                    ]
    
    def _save(self):
        """Save registry to disk."""
        data = {}
        for model_id, versions in self.models.items():
            data[model_id] = [asdict(v) for v in versions]
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_checksum(self, params: Dict) -> str:
        """Compute checksum of model parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:12]
    
    def register(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        parameters: Dict,
        status: str = 'staging'
    ) -> ModelVersion:
        """Register a new model version."""
        if model_id not in self.models:
            self.models[model_id] = []
        
        # Auto-increment version
        version_num = len(self.models[model_id]) + 1
        version = f"v{version_num}.0"
        
        model = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            parameters=parameters,
            status=status,
            checksum=self._compute_checksum(parameters)
        )
        
        self.models[model_id].append(model)
        self._save()
        return model
    
    def promote_to_production(self, model_id: str, version: str):
        """Promote a model to production."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        for model in self.models[model_id]:
            if model.status == 'production':
                model.status = 'archived'
            if model.version == version:
                model.status = 'production'
        
        self._save()
    
    def rollback(self, model_id: str) -> Optional[ModelVersion]:
        """Rollback to previous production version."""
        if model_id not in self.models:
            return None
        
        versions = self.models[model_id]
        archived = [v for v in versions if v.status == 'archived']
        
        if not archived:
            return None
        
        # Find current production
        for v in versions:
            if v.status == 'production':
                v.status = 'archived'
        
        # Promote most recent archived
        last_archived = archived[-1]
        last_archived.status = 'production'
        
        self._save()
        return last_archived
    
    def get_production(self, model_id: str) -> Optional[ModelVersion]:
        """Get current production model."""
        if model_id not in self.models:
            return None
        
        for v in self.models[model_id]:
            if v.status == 'production':
                return v
        return None
    
    def list_versions(self, model_id: str) -> List[Dict]:
        """List all versions of a model."""
        if model_id not in self.models:
            return []
        return [asdict(v) for v in self.models[model_id]]
    
    def compare_versions(self, model_id: str, v1: str, v2: str) -> Dict:
        """Compare two model versions."""
        versions = {v.version: v for v in self.models.get(model_id, [])}
        
        if v1 not in versions or v2 not in versions:
            return {'error': 'Version not found'}
        
        m1, m2 = versions[v1], versions[v2]
        
        metric_diff = {}
        for metric in set(m1.metrics.keys()) | set(m2.metrics.keys()):
            val1 = m1.metrics.get(metric, 0)
            val2 = m2.metrics.get(metric, 0)
            metric_diff[metric] = {
                'v1': val1,
                'v2': val2,
                'diff': val2 - val1,
                'pct_change': (val2 - val1) / (val1 + 1e-8) * 100
            }
        
        return {
            'v1': v1,
            'v2': v2,
            'metric_comparison': metric_diff,
            'param_changes': {
                k: {'v1': m1.parameters.get(k), 'v2': m2.parameters.get(k)}
                for k in set(m1.parameters.keys()) | set(m2.parameters.keys())
                if m1.parameters.get(k) != m2.parameters.get(k)
            }
        }
