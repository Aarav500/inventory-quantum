"""ABC Analysis router for Pareto-based inventory categorization."""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import random

router = APIRouter()


class ABCItem(BaseModel):
    """Single item with ABC classification."""
    sku: str
    name: str
    category: str  # A, B, or C
    annual_value: float
    annual_quantity: int
    unit_cost: float
    cumulative_percentage: float
    rank: int


class ABCAnalysisResult(BaseModel):
    """Complete ABC analysis result."""
    analysis_date: str
    total_items: int
    total_value: float
    category_a: dict
    category_b: dict
    category_c: dict
    items: List[ABCItem]


class ABCRecommendation(BaseModel):
    """Recommendations based on ABC category."""
    category: str
    item_count: int
    value_percentage: float
    recommendations: List[str]
    reorder_strategy: str
    safety_stock_policy: str
    review_frequency: str


def generate_demo_abc_data(item_count: int = 100) -> List[ABCItem]:
    """Generate demo ABC analysis data following Pareto distribution."""
    items = []
    
    # Generate items with Pareto-like distribution
    for i in range(item_count):
        sku = f"SKU-{str(i+1).zfill(4)}"
        # Higher values for first items (A category)
        if i < int(item_count * 0.2):  # Top 20%
            annual_value = random.uniform(50000, 200000)
        elif i < int(item_count * 0.5):  # Next 30%
            annual_value = random.uniform(10000, 50000)
        else:  # Bottom 50%
            annual_value = random.uniform(500, 10000)
        
        unit_cost = random.uniform(10, 500)
        annual_quantity = int(annual_value / unit_cost)
        
        items.append({
            "sku": sku,
            "name": f"Product {i+1}",
            "annual_value": annual_value,
            "annual_quantity": annual_quantity,
            "unit_cost": unit_cost
        })
    
    # Sort by annual value descending
    items.sort(key=lambda x: x["annual_value"], reverse=True)
    
    # Calculate cumulative percentages and assign categories
    total_value = sum(item["annual_value"] for item in items)
    cumulative = 0
    result = []
    
    for rank, item in enumerate(items, 1):
        cumulative += item["annual_value"]
        cumulative_pct = (cumulative / total_value) * 100
        
        # Assign category based on cumulative value
        if cumulative_pct <= 80:
            category = "A"
        elif cumulative_pct <= 95:
            category = "B"
        else:
            category = "C"
        
        result.append(ABCItem(
            sku=item["sku"],
            name=item["name"],
            category=category,
            annual_value=round(item["annual_value"], 2),
            annual_quantity=item["annual_quantity"],
            unit_cost=round(item["unit_cost"], 2),
            cumulative_percentage=round(cumulative_pct, 2),
            rank=rank
        ))
    
    return result


@router.post("/analyze", response_model=ABCAnalysisResult)
async def run_abc_analysis(
    use_demo_data: bool = Query(True, description="Use demo data for analysis")
):
    """
    Run ABC (Pareto) analysis on inventory.
    
    Classifies items into A (high value), B (medium), C (low) categories
    based on the 80/20 rule.
    """
    items = generate_demo_abc_data(100)
    
    # Calculate category statistics
    category_a_items = [i for i in items if i.category == "A"]
    category_b_items = [i for i in items if i.category == "B"]
    category_c_items = [i for i in items if i.category == "C"]
    
    total_value = sum(i.annual_value for i in items)
    
    return ABCAnalysisResult(
        analysis_date=datetime.now().isoformat(),
        total_items=len(items),
        total_value=round(total_value, 2),
        category_a={
            "count": len(category_a_items),
            "percentage_of_items": round(len(category_a_items) / len(items) * 100, 1),
            "value": round(sum(i.annual_value for i in category_a_items), 2),
            "percentage_of_value": round(sum(i.annual_value for i in category_a_items) / total_value * 100, 1)
        },
        category_b={
            "count": len(category_b_items),
            "percentage_of_items": round(len(category_b_items) / len(items) * 100, 1),
            "value": round(sum(i.annual_value for i in category_b_items), 2),
            "percentage_of_value": round(sum(i.annual_value for i in category_b_items) / total_value * 100, 1)
        },
        category_c={
            "count": len(category_c_items),
            "percentage_of_items": round(len(category_c_items) / len(items) * 100, 1),
            "value": round(sum(i.annual_value for i in category_c_items), 2),
            "percentage_of_value": round(sum(i.annual_value for i in category_c_items) / total_value * 100, 1)
        },
        items=items
    )


@router.get("/results", response_model=List[ABCItem])
async def get_abc_results(
    category: Optional[str] = Query(None, description="Filter by category (A, B, C)"),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get ABC classification results.
    
    Returns items with their ABC category assignments.
    """
    items = generate_demo_abc_data(100)
    
    if category:
        items = [i for i in items if i.category.upper() == category.upper()]
    
    return items[:limit]


@router.get("/recommendations", response_model=List[ABCRecommendation])
async def get_abc_recommendations():
    """
    Get management recommendations based on ABC categories.
    
    Provides actionable strategies for each inventory category.
    """
    items = generate_demo_abc_data(100)
    total_value = sum(i.annual_value for i in items)
    
    recommendations = [
        ABCRecommendation(
            category="A",
            item_count=len([i for i in items if i.category == "A"]),
            value_percentage=round(sum(i.annual_value for i in items if i.category == "A") / total_value * 100, 1),
            recommendations=[
                "Implement tight inventory control with frequent reviews",
                "Negotiate better terms with suppliers",
                "Use accurate demand forecasting models",
                "Maintain detailed transaction records",
                "Consider vendor-managed inventory (VMI)"
            ],
            reorder_strategy="Use EOQ with safety stock, frequent small orders",
            safety_stock_policy="Higher safety stock levels (2-3 weeks coverage)",
            review_frequency="Daily to weekly"
        ),
        ABCRecommendation(
            category="B",
            item_count=len([i for i in items if i.category == "B"]),
            value_percentage=round(sum(i.annual_value for i in items if i.category == "B") / total_value * 100, 1),
            recommendations=[
                "Moderate control with periodic reviews",
                "Use standard forecasting methods",
                "Balance between ordering costs and holding costs",
                "Group similar items for bulk ordering"
            ],
            reorder_strategy="Fixed order intervals with variable quantities",
            safety_stock_policy="Moderate safety stock (1-2 weeks coverage)",
            review_frequency="Weekly to bi-weekly"
        ),
        ABCRecommendation(
            category="C",
            item_count=len([i for i in items if i.category == "C"]),
            value_percentage=round(sum(i.annual_value for i in items if i.category == "C") / total_value * 100, 1),
            recommendations=[
                "Simplify control procedures",
                "Order in large quantities to reduce ordering costs",
                "Consider consignment or dropship arrangements",
                "Use simple min-max replenishment"
            ],
            reorder_strategy="Infrequent large orders, min-max system",
            safety_stock_policy="Minimal safety stock (0.5-1 week coverage)",
            review_frequency="Monthly or quarterly"
        )
    ]
    
    return recommendations


@router.get("/pareto-data")
async def get_pareto_chart_data():
    """
    Get data formatted for Pareto chart visualization.
    
    Returns cumulative percentages for chart plotting.
    """
    items = generate_demo_abc_data(100)
    
    chart_data = {
        "labels": [i.sku for i in items[:50]],  # Top 50 for chart
        "values": [i.annual_value for i in items[:50]],
        "cumulative": [i.cumulative_percentage for i in items[:50]],
        "categories": [i.category for i in items[:50]],
        "summary": {
            "a_threshold": 80,
            "b_threshold": 95,
            "total_items": len(items),
            "category_counts": {
                "A": len([i for i in items if i.category == "A"]),
                "B": len([i for i in items if i.category == "B"]),
                "C": len([i for i in items if i.category == "C"])
            }
        }
    }
    
    return chart_data
