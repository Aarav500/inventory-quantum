"""
Natural Language Query Interface.

Allows users to ask questions about inventory in plain English:
- "What is the forecast for SKU001 next week?"
- "Why did we have a stockout last month?"
- "Optimize inventory for 95% service level"
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Parsed query intent."""
    action: str
    entity: Optional[str]
    parameters: Dict
    confidence: float


class NLQueryParser:
    """
    Parse natural language queries into structured intents.
    
    Uses rule-based pattern matching (production would use LLM).
    """
    
    def __init__(self):
        self.patterns = {
            'forecast': [
                r'(?:what is|show|get|predict)\s+(?:the\s+)?forecast\s+(?:for\s+)?(\w+)',
                r'forecast\s+(?:for\s+)?(\w+)',
                r'predict\s+demand\s+(?:for\s+)?(\w+)',
            ],
            'explain': [
                r'(?:why|explain|what caused)\s+.*(stockout|spike|drop)',
                r'explain\s+(?:the\s+)?(\w+)',
            ],
            'optimize': [
                r'optimize\s+(?:inventory\s+)?(?:for\s+)?(\d+)%?\s*(?:service)?',
                r'find\s+optimal\s+(\w+)',
                r'what\s+should\s+(?:the\s+)?(\w+)\s+be',
            ],
            'compare': [
                r'compare\s+(\w+)\s+(?:to|with|vs)\s+(\w+)',
                r'which\s+is\s+better.*(model|method)',
            ],
            'status': [
                r'(?:what is|show)\s+(?:the\s+)?status\s+(?:of\s+)?(\w+)',
                r'(?:current|today).*inventory',
            ],
            'risk': [
                r'(?:what is|show)\s+(?:the\s+)?risk',
                r'stockout\s+(?:risk|probability)',
            ],
            'causal': [
                r'(?:what is|show)\s+(?:the\s+)?(?:effect|impact)\s+of\s+(\w+)',
                r'does\s+(\w+)\s+(?:affect|impact)\s+(\w+)',
            ]
        }
    
    def parse(self, query: str) -> QueryIntent:
        """Parse natural language query."""
        query = query.lower().strip()
        
        for action, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    return QueryIntent(
                        action=action,
                        entity=match.group(1) if match.groups() else None,
                        parameters=self._extract_parameters(query),
                        confidence=0.8
                    )
        
        # Default/fallback
        return QueryIntent(
            action='unknown',
            entity=None,
            parameters={'raw_query': query},
            confidence=0.3
        )
    
    def _extract_parameters(self, query: str) -> Dict:
        """Extract parameters from query."""
        params = {}
        
        # Time periods
        if 'week' in query:
            params['horizon'] = 7
        elif 'month' in query:
            params['horizon'] = 30
        elif 'quarter' in query:
            params['horizon'] = 90
        
        # Percentages
        pct_match = re.search(r'(\d+)%', query)
        if pct_match:
            params['percentage'] = int(pct_match.group(1))
        
        # SKU identifiers
        sku_match = re.search(r'sku[-_]?(\w+)', query, re.IGNORECASE)
        if sku_match:
            params['sku'] = f"SKU{sku_match.group(1)}"
        
        return params


class NLQueryExecutor:
    """
    Execute parsed queries against the inventory system.
    """
    
    def __init__(self, data_context: Dict = None):
        self.data = data_context or {}
        self.parser = NLQueryParser()
    
    def execute(self, query: str) -> Dict:
        """Execute natural language query."""
        intent = self.parser.parse(query)
        
        handlers = {
            'forecast': self._handle_forecast,
            'explain': self._handle_explain,
            'optimize': self._handle_optimize,
            'compare': self._handle_compare,
            'status': self._handle_status,
            'risk': self._handle_risk,
            'causal': self._handle_causal,
            'unknown': self._handle_unknown,
        }
        
        handler = handlers.get(intent.action, self._handle_unknown)
        result = handler(intent)
        
        return {
            'query': query,
            'intent': {
                'action': intent.action,
                'entity': intent.entity,
                'parameters': intent.parameters,
                'confidence': intent.confidence
            },
            'result': result
        }
    
    def _handle_forecast(self, intent: QueryIntent) -> Dict:
        """Handle forecast queries."""
        sku = intent.entity or intent.parameters.get('sku', 'ALL')
        horizon = intent.parameters.get('horizon', 7)
        
        # Simulated response (real system would call forecaster)
        return {
            'type': 'forecast',
            'sku': sku,
            'horizon_days': horizon,
            'predictions': [50 + i * 0.5 for i in range(horizon)],
            'confidence_interval': {
                'lower': [45 + i * 0.5 for i in range(horizon)],
                'upper': [55 + i * 0.5 for i in range(horizon)]
            },
            'narrative': f"Forecast for {sku} shows stable demand around 50-55 units/day over the next {horizon} days."
        }
    
    def _handle_explain(self, intent: QueryIntent) -> Dict:
        """Handle explanation queries."""
        event = intent.entity or 'anomaly'
        
        return {
            'type': 'explanation',
            'event': event,
            'factors': [
                {'name': 'Promotion ended', 'contribution': 0.35},
                {'name': 'Seasonality', 'contribution': 0.25},
                {'name': 'Supplier delay', 'contribution': 0.20},
                {'name': 'Competition', 'contribution': 0.15},
                {'name': 'Other', 'contribution': 0.05}
            ],
            'narrative': f"The {event} was primarily caused by promotion ending (35%) and seasonal decline (25%)."
        }
    
    def _handle_optimize(self, intent: QueryIntent) -> Dict:
        """Handle optimization queries."""
        service_level = intent.parameters.get('percentage', 95)
        
        return {
            'type': 'optimization',
            'target_service_level': service_level,
            'recommendations': {
                'reorder_point': 120,
                'order_quantity': 85,
                'expected_cost': 2340,
                'achieved_service_level': 95.2
            },
            'narrative': f"To achieve {service_level}% service level, set reorder point to 120 and order quantity to 85."
        }
    
    def _handle_compare(self, intent: QueryIntent) -> Dict:
        """Handle comparison queries."""
        return {
            'type': 'comparison',
            'models': ['TFT', 'LightGBM', 'ARIMA'],
            'metrics': {
                'TFT': {'rmse': 4.2, 'mape': 9.1},
                'LightGBM': {'rmse': 5.1, 'mape': 11.3},
                'ARIMA': {'rmse': 6.8, 'mape': 14.2}
            },
            'winner': 'TFT',
            'narrative': "TFT outperforms other models with 4.2 RMSE and 9.1% MAPE."
        }
    
    def _handle_status(self, intent: QueryIntent) -> Dict:
        """Handle status queries."""
        sku = intent.entity or 'ALL'
        
        return {
            'type': 'status',
            'sku': sku,
            'current_inventory': 145,
            'reorder_point': 100,
            'status': 'healthy',
            'days_of_stock': 8,
            'narrative': f"{sku} has 145 units in stock (8 days supply). No action needed."
        }
    
    def _handle_risk(self, intent: QueryIntent) -> Dict:
        """Handle risk queries."""
        return {
            'type': 'risk',
            'stockout_probability': 0.12,
            'expected_shortfall': 15,
            'var_95': 2800,
            'narrative': "12% probability of stockout in next 7 days. Expected shortfall: 15 units."
        }
    
    def _handle_causal(self, intent: QueryIntent) -> Dict:
        """Handle causal inference queries."""
        treatment = intent.entity or 'promotion'
        
        return {
            'type': 'causal',
            'treatment': treatment,
            'effect': 18.5,
            'confidence_interval': [12.3, 24.7],
            'p_value': 0.003,
            'narrative': f"Promotions causally increase demand by 18.5 units (95% CI: 12.3-24.7, p=0.003)."
        }
    
    def _handle_unknown(self, intent: QueryIntent) -> Dict:
        """Handle unknown queries."""
        return {
            'type': 'unknown',
            'suggestions': [
                "Try: 'What is the forecast for SKU001?'",
                "Try: 'Optimize for 95% service level'",
                "Try: 'Why did we have a stockout?'",
                "Try: 'What is the effect of promotions?'"
            ],
            'narrative': "I didn't understand that query. Here are some examples you can try."
        }
