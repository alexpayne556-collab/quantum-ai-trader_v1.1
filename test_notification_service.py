"""
Test Notification Service with real Discord webhook
"""
import asyncio
from datetime import datetime
from notification_service import NotificationService, Alert

async def test_notification_service():
    print("🚀 Testing Multi-Channel Notification Service...")
    print("="*80)
    
    service = NotificationService()
    
    # Test Alert 1: HIGH priority STRONG_BUY
    alert1 = Alert(
        ticker='AAPL',
        signal='STRONG_BUY',
        price=278.85,
        signal_strength=85.3,
        rsi_5m=35.2,
        rsi_1h=42.8,
        macd_signal='BULLISH',
        momentum=68.5,
        timestamp=datetime.now(),
        priority='HIGH',
        channel=['discord']  # Test Discord only
    )
    
    print("\n📨 Sending HIGH priority STRONG_BUY alert...")
    results1 = await service.send_alert(alert1)
    print(f"Results: {results1}")
    
    # Test Alert 2: MEDIUM priority BUY
    alert2 = Alert(
        ticker='NVDA',
        signal='BUY',
        price=177.00,
        signal_strength=72.1,
        rsi_5m=55.3,
        rsi_1h=58.7,
        macd_signal='BULLISH',
        momentum=62.3,
        timestamp=datetime.now(),
        priority='MEDIUM',
        channel=['discord']
    )
    
    print("\n📨 Sending MEDIUM priority BUY alert...")
    results2 = await service.send_alert(alert2)
    print(f"Results: {results2}")
    
    # Test Alert 3: Rate limiter test (same ticker)
    alert3 = Alert(
        ticker='AAPL',
        signal='BUY',
        price=279.15,
        signal_strength=68.5,
        rsi_5m=45.2,
        rsi_1h=48.3,
        macd_signal='BULLISH',
        momentum=55.2,
        timestamp=datetime.now(),
        priority='MEDIUM',
        channel=['discord']
    )
    
    print("\n📨 Testing rate limiter (should allow first few, then block)...")
    for i in range(5):
        results = await service.send_alert(alert3)
        print(f"Attempt {i+1}: {results if results else 'BLOCKED by rate limiter'}")
        await asyncio.sleep(0.5)
    
    print("\n" + "="*80)
    print("✅ Notification service test completed!")
    print("\nNote: Configure environment variables for Email/SMS:")
    print("  - DISCORD_WEBHOOK (for Discord)")
    print("  - EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO (for Email)")
    print("  - TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, etc. (for SMS)")

if __name__ == "__main__":
    asyncio.run(test_notification_service())
