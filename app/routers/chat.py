"""AI Chat Assistant router for natural language inventory queries."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import random

from app.routers.upload import get_data

router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]
    data: Optional[dict] = None


def get_insights():
    """Generate insights from uploaded data."""
    df = get_data()
    
    # Default Demo Data
    insights = {
        "stockout": {
            "items_at_risk": ["SKU-003", "SKU-007", "SKU-012"],
            "days_until_stockout": {"SKU-003": 5, "SKU-007": 8, "SKU-012": 3},
        },
        "top_sellers": ["SKU-001", "SKU-005", "SKU-008"],
        "slow_movers": ["SKU-015", "SKU-022", "SKU-031"],
        "total_skus": 150,
        "total_value": 2450000,
        "avg_turnover": 4.2,
        "source": "Demo Data"
    }
    
    if df is not None:
        try:
            # 1. Total SKUs
            skus = df['sku'].unique()
            insights["total_skus"] = len(skus)
            
            # 2. Total Value
            val = 0
            if 'price' in df.columns and 'quantity_on_hand' in df.columns:
                 # Estimate from latest snapshot
                 latest = df.sort_values('date').groupby('sku').last()
                 val = (latest['quantity_on_hand'] * latest['price']).sum()
            elif 'quantity_sold' in df.columns:
                 # Rough estimate
                 val = df['quantity_sold'].sum() * 50 # avg price
            
            insights["total_value"] = int(val)
            
            # 3. Top Sellers (by qty sold)
            top = df.groupby('sku')['quantity_sold'].sum().sort_values(ascending=False).head(3)
            insights["top_sellers"] = top.index.tolist()
            
            # 4. Slow Movers
            bottom = df.groupby('sku')['quantity_sold'].sum().sort_values(ascending=True).head(3)
            insights["slow_movers"] = bottom.index.tolist()
            
            # 5. Stockout Risk
            at_risk = []
            days_map = {}
            for sku in skus:
                sku_data = df[df['sku'] == sku]
                avg = sku_data['quantity_sold'].mean()
                curr = 0
                if 'quantity_on_hand' in sku_data.columns:
                    curr = sku_data['quantity_on_hand'].iloc[-1] if not sku_data['quantity_on_hand'].isna().all() else 0
                
                days = int(curr / avg) if avg > 0 else 999
                if days < 10:
                    at_risk.append(str(sku))
                    days_map[str(sku)] = days
            
            insights["stockout"]["items_at_risk"] = at_risk[:5]
            insights["stockout"]["days_until_stockout"] = {k: days_map[k] for k in at_risk[:5]}
            insights["source"] = "Your Uploaded Data"
            
        except Exception as e:
            print(f"Error generating chat insights: {e}")
            
    return insights


def process_message(message: str) -> ChatResponse:
    """Process user message and generate response."""
    msg_lower = message.lower()
    insights = get_insights()
    
    # Reorder/stockout queries
    if any(word in msg_lower for word in ["reorder", "stock out", "stockout", "running low", "order"]):
        items = insights["stockout"]["items_at_risk"]
        days = insights["stockout"]["days_until_stockout"]
        
        if not items:
            return ChatResponse(
                response="✅ **Great news!**\n\nNo items are currently at immediate risk of stocking out based on your data.",
                suggestions=["Show inventory summary", "View top sellers"],
                data={"status": "healthy"}
            )
            
        return ChatResponse(
            response=f"📦 **Items to Reorder Soon ({insights['source']}):**\n\n" + 
                    "\n".join([f"• **{sku}** - {days.get(sku, '?')} days left" for sku in items]) +
                    "\n\nI recommend reviewing these items immediately.",
            suggestions=["Show me the full stockout report", "What's the recommended order quantity?"],
            data={"at_risk_items": items, "days": days}
        )
    
    # Top sellers
    elif any(word in msg_lower for word in ["top", "best", "selling", "popular", "fast"]):
        items = insights["top_sellers"]
        return ChatResponse(
            response=f"🏆 **Top Selling Items ({insights['source']}):**\n\n" +
                    "\n".join([f"• **{sku}**" for sku in items]) +
                    "\n\nThese items are your volume leaders.",
            suggestions=["Which supplier provides these?", "Show demand forecast"],
            data={"top_sellers": items}
        )
    
    # Summary
    elif any(word in msg_lower for word in ["summary", "overview", "total", "how many", "status"]):
        return ChatResponse(
            response=f"📊 **Inventory Summary ({insights['source']}):**\n\n" +
                    f"• **Total SKUs:** {insights['total_skus']}\n" +
                    f"• **Total Value:** ${insights['total_value']:,}\n" +
                    f"• **Items at Risk:** {len(insights['stockout']['items_at_risk'])}",
            suggestions=["Break down by category", "Show top sellers"],
            data=insights
        )

    # Cost
    elif any(word in msg_lower for word in ["cost", "save", "money"]):
        return ChatResponse(
             response=f"💰 **Cost Optimization:**\n\n" +
                      f"Based on your {insights['total_skus']} SKUs, we've identified potential savings in holding costs.\n" +
                      "Check the Cost Savings Calculator for a detailed report.",
             suggestions=["Go to Savings Calculator"],
             data=None
        )

    # Help/greeting
    elif any(word in msg_lower for word in ["help", "hello", "hi", "hey"]):
        return ChatResponse(
            response=f"👋 **Hi! I'm your Inventory Assistant.**\n\n" +
                    f"I'm analyzing **{insights['source']}**.\n\n" +
                    "Ask me about:\n" +
                    "• 📦 Reorder needs\n" +
                    "• 🏆 Top sellers\n" +
                    "• 📊 Inventory summary",
            suggestions=["What needs reordering?", "Show summary"],
            data=None
        )
    
    # Default
    else:
        return ChatResponse(
            response="🤔 I'm not sure I understand. Try asking about reorders, top sellers, or an inventory summary.",
            suggestions=["What needs reordering?", "Show summary"],
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
            "Give me an inventory summary",
        ]
    }
