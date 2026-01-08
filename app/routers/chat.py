"""AI Chat Assistant router for natural language inventory queries."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import random

router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]
    data: Optional[dict] = None


# Knowledge base for inventory insights
INVENTORY_INSIGHTS = {
    "stockout": {
        "items_at_risk": ["SKU-003", "SKU-007", "SKU-012"],
        "days_until_stockout": {"SKU-003": 5, "SKU-007": 8, "SKU-012": 3},
    },
    "top_sellers": ["SKU-001", "SKU-005", "SKU-008"],
    "slow_movers": ["SKU-015", "SKU-022", "SKU-031"],
    "total_skus": 150,
    "total_value": 2450000,
    "avg_turnover": 4.2,
}


def process_message(message: str) -> ChatResponse:
    """Process user message and generate response."""
    msg_lower = message.lower()
    
    # Reorder/stockout queries
    if any(word in msg_lower for word in ["reorder", "stock out", "stockout", "running low", "order"]):
        items = INVENTORY_INSIGHTS["stockout"]["items_at_risk"]
        days = INVENTORY_INSIGHTS["stockout"]["days_until_stockout"]
        return ChatResponse(
            response=f"📦 **Items to Reorder Soon:**\n\n" + 
                    "\n".join([f"• **{sku}** - {days[sku]} days until stockout" for sku in items]) +
                    "\n\nI recommend placing orders for these items within the next 48 hours to avoid stockouts.",
            suggestions=["Show me the full stockout report", "What's the recommended order quantity?", "Which supplier is fastest?"],
            data={"at_risk_items": items, "days_until_stockout": days}
        )
    
    # Top sellers
    elif any(word in msg_lower for word in ["top", "best", "selling", "popular", "fast"]):
        items = INVENTORY_INSIGHTS["top_sellers"]
        return ChatResponse(
            response=f"🏆 **Top Selling Items:**\n\n" +
                    "\n".join([f"• **{sku}**" for sku in items]) +
                    "\n\nThese items have the highest demand velocity. Consider increasing safety stock.",
            suggestions=["What's driving sales for these?", "Show demand forecast", "Optimize reorder points"],
            data={"top_sellers": items}
        )
    
    # Slow movers
    elif any(word in msg_lower for word in ["slow", "dead", "excess", "overstock"]):
        items = INVENTORY_INSIGHTS["slow_movers"]
        return ChatResponse(
            response=f"🐌 **Slow Moving Items:**\n\n" +
                    "\n".join([f"• **{sku}**" for sku in items]) +
                    "\n\nConsider running promotions or reducing order quantities for these items.",
            suggestions=["Calculate holding costs", "Suggest markdown pricing", "Transfer to other locations"],
            data={"slow_movers": items}
        )
    
    # Summary/overview
    elif any(word in msg_lower for word in ["summary", "overview", "total", "how many", "status"]):
        return ChatResponse(
            response=f"📊 **Inventory Summary:**\n\n" +
                    f"• **Total SKUs:** {INVENTORY_INSIGHTS['total_skus']}\n" +
                    f"• **Total Value:** ${INVENTORY_INSIGHTS['total_value']:,}\n" +
                    f"• **Avg Turnover:** {INVENTORY_INSIGHTS['avg_turnover']}x/year\n" +
                    f"• **Items at Risk:** {len(INVENTORY_INSIGHTS['stockout']['items_at_risk'])}",
            suggestions=["Break down by category", "Show value by location", "Compare to last month"],
            data=INVENTORY_INSIGHTS
        )
    
    # Forecast
    elif any(word in msg_lower for word in ["forecast", "predict", "demand", "next week", "next month"]):
        return ChatResponse(
            response="📈 **Demand Forecast Summary:**\n\n" +
                    "• **Next 7 days:** +12% expected increase\n" +
                    "• **Next 30 days:** Stable with slight upward trend\n" +
                    "• **Seasonal impact:** Holiday season approaching, expect 25% surge\n\n" +
                    "Go to the **Forecasting** page for detailed predictions.",
            suggestions=["Show forecast chart", "Which items will spike?", "Update forecast model"],
            data={"trend": "increasing", "confidence": 0.89}
        )
    
    # Savings/costs
    elif any(word in msg_lower for word in ["save", "cost", "money", "reduce", "optimize"]):
        savings = random.randint(15000, 45000)
        return ChatResponse(
            response=f"💰 **Cost Optimization Insights:**\n\n" +
                    f"• **Potential Monthly Savings:** ${savings:,}\n" +
                    "• **Holding Cost Reduction:** 15% possible\n" +
                    "• **Stockout Cost Avoidance:** $8,500/month\n\n" +
                    "Visit the **Cost Savings Calculator** for detailed analysis.",
            suggestions=["Run full optimization", "Show ROI breakdown", "Compare scenarios"],
            data={"potential_savings": savings}
        )
    
    # Help/greeting
    elif any(word in msg_lower for word in ["help", "hello", "hi", "hey", "what can"]):
        return ChatResponse(
            response="👋 **Hi! I'm your Inventory Assistant.**\n\n" +
                    "I can help you with:\n" +
                    "• 📦 **Reorder recommendations** - What needs restocking?\n" +
                    "• 📈 **Demand forecasts** - What's coming up?\n" +
                    "• 🏆 **Top/slow movers** - Best and worst performers\n" +
                    "• 💰 **Cost optimization** - Where can you save?\n" +
                    "• 📊 **Inventory summary** - Overall status\n\n" +
                    "Just ask me a question!",
            suggestions=["What should I reorder?", "Show inventory summary", "How can I reduce costs?"],
            data=None
        )
    
    # Default
    else:
        return ChatResponse(
            response="🤔 I'm not sure I understand. Try asking about:\n\n" +
                    "• What items need reordering?\n" +
                    "• Show me top selling products\n" +
                    "• What's my inventory summary?\n" +
                    "• How can I reduce costs?",
            suggestions=["What should I reorder?", "Show top sellers", "Give me a summary"],
            data=None
        )


@router.post("/message", response_model=ChatResponse)
async def chat_message(chat: ChatMessage):
    """Process a chat message and return AI response."""
    return process_message(chat.message)


@router.get("/suggestions")
async def get_suggestions():
    """Get suggested questions for the chat."""
    return {
        "suggestions": [
            "What items should I reorder?",
            "Show me top selling products",
            "Which items are slow moving?",
            "Give me an inventory summary",
            "What's the demand forecast?",
            "How can I reduce costs?"
        ]
    }
