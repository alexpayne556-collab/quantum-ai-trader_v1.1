"""
Notification & Alert Module
Research-backed: Multi-channel alerts (Email, SMS, Webhooks, Discord)
Priority routing, rate limiting, and real-time signal distribution

Based on research.md best practices for small account (<$1k) trading
"""
import asyncio
import aiohttp
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import Config

@dataclass
class Alert:
    """Trading alert structure"""
    ticker: str
    signal: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    price: float
    signal_strength: float
    rsi_5m: float
    rsi_1h: float
    macd_signal: str
    momentum: float
    timestamp: datetime
    priority: str  # HIGH, MEDIUM, LOW
    channel: List[str]  # email, sms, webhook, discord

class RateLimiter:
    """Rate limiter to prevent alert spam"""
    
    def __init__(self, max_alerts_per_hour: int = 10, max_alerts_per_symbol: int = 3):
        self.max_alerts_per_hour = max_alerts_per_hour
        self.max_alerts_per_symbol = max_alerts_per_symbol
        self.alert_history: Dict[str, List[datetime]] = defaultdict(list)
        self.symbol_alerts: Dict[str, List[datetime]] = defaultdict(list)
    
    def can_send_alert(self, symbol: str) -> bool:
        """Check if alert can be sent based on rate limits"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean old alerts
        self.alert_history['global'] = [
            ts for ts in self.alert_history['global'] if ts > one_hour_ago
        ]
        self.symbol_alerts[symbol] = [
            ts for ts in self.symbol_alerts[symbol] if ts > one_hour_ago
        ]
        
        # Check limits
        if len(self.alert_history['global']) >= self.max_alerts_per_hour:
            return False
        
        if len(self.symbol_alerts[symbol]) >= self.max_alerts_per_symbol:
            return False
        
        return True
    
    def record_alert(self, symbol: str):
        """Record alert sent"""
        now = datetime.now()
        self.alert_history['global'].append(now)
        self.symbol_alerts[symbol].append(now)

class NotificationService:
    """Multi-channel notification service with priority routing"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(max_alerts_per_hour=10, max_alerts_per_symbol=3)
        self.alert_queue: List[Alert] = []
        self.callbacks: List[Callable] = []
        
        # Email config (using Gmail SMTP as example)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_from = os.getenv("EMAIL_FROM", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.email_to = os.getenv("EMAIL_TO", "")
        
        # Discord webhook
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK", "")
        
        # Twilio (SMS) config
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_from = os.getenv("TWILIO_FROM", "")
        self.twilio_to = os.getenv("TWILIO_TO", "")
    
    def prioritize_alert(self, alert: Alert) -> str:
        """Determine alert priority based on signal strength and conditions"""
        # HIGH: Strong signals with high momentum
        if alert.signal in ['STRONG_BUY', 'STRONG_SELL']:
            if alert.signal_strength >= 75 and alert.momentum >= 55:
                return 'HIGH'
        
        # MEDIUM: Good signals with decent strength
        if alert.signal in ['BUY', 'SELL']:
            if alert.signal_strength >= 65:
                return 'MEDIUM'
        
        # LOW: Everything else
        return 'LOW'
    
    async def send_email(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.email_from or not self.email_to:
            print("⚠️ Email not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"🚨 [{alert.priority}] {alert.ticker} - {alert.signal}"
            
            body = f"""
Trading Alert - {alert.ticker}

Signal: {alert.signal}
Price: ${alert.price:.2f}
Signal Strength: {alert.signal_strength:.1f}/100
Priority: {alert.priority}

Technical Indicators:
- RSI (5m): {alert.rsi_5m:.1f}
- RSI (1h): {alert.rsi_1h:.1f}
- MACD: {alert.macd_signal}
- Momentum: {alert.momentum:.1f}/100

Time: {alert.timestamp}

Action: {"ENTER POSITION" if alert.signal in ['BUY', 'STRONG_BUY'] else "EXIT POSITION" if alert.signal in ['SELL', 'STRONG_SELL'] else "MONITOR"}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send via SMTP
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_from, self.email_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent for {alert.ticker}")
            return True
            
        except Exception as e:
            print(f"❌ Email failed: {e}")
            return False
    
    async def send_discord(self, alert: Alert) -> bool:
        """Send Discord webhook notification"""
        if not self.discord_webhook:
            print("⚠️ Discord webhook not configured")
            return False
        
        try:
            # Color based on signal
            color = {
                'STRONG_BUY': 0x00FF00,  # Green
                'BUY': 0x90EE90,  # Light green
                'HOLD': 0xFFFF00,  # Yellow
                'SELL': 0xFFA500,  # Orange
                'STRONG_SELL': 0xFF0000  # Red
            }.get(alert.signal, 0x808080)
            
            embed = {
                "title": f"🚨 {alert.ticker} - {alert.signal}",
                "description": f"Signal Strength: {alert.signal_strength:.1f}/100",
                "color": color,
                "fields": [
                    {"name": "Price", "value": f"${alert.price:.2f}", "inline": True},
                    {"name": "Priority", "value": alert.priority, "inline": True},
                    {"name": "RSI (5m/1h)", "value": f"{alert.rsi_5m:.1f} / {alert.rsi_1h:.1f}", "inline": True},
                    {"name": "MACD", "value": alert.macd_signal, "inline": True},
                    {"name": "Momentum", "value": f"{alert.momentum:.1f}/100", "inline": True},
                ],
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "Quantum AI Trader v1.1"}
            }
            
            payload = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as response:
                    if response.status == 204:
                        print(f"✅ Discord webhook sent for {alert.ticker}")
                        return True
                    else:
                        print(f"❌ Discord webhook failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Discord webhook failed: {e}")
            return False
    
    async def send_sms(self, alert: Alert) -> bool:
        """Send SMS via Twilio"""
        if not self.twilio_account_sid or not self.twilio_to:
            print("⚠️ Twilio not configured")
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            message_body = f"""
🚨 {alert.ticker} - {alert.signal}
Price: ${alert.price:.2f}
Strength: {alert.signal_strength:.0f}/100
RSI: {alert.rsi_5m:.0f}
Priority: {alert.priority}
            """
            
            message = client.messages.create(
                body=message_body.strip(),
                from_=self.twilio_from,
                to=self.twilio_to
            )
            
            print(f"✅ SMS sent for {alert.ticker}: {message.sid}")
            return True
            
        except Exception as e:
            print(f"❌ SMS failed: {e}")
            return False
    
    async def send_webhook(self, alert: Alert, webhook_url: str) -> bool:
        """Send custom webhook notification"""
        try:
            payload = {
                'ticker': alert.ticker,
                'signal': alert.signal,
                'price': alert.price,
                'signal_strength': alert.signal_strength,
                'rsi_5m': alert.rsi_5m,
                'rsi_1h': alert.rsi_1h,
                'macd_signal': alert.macd_signal,
                'momentum': alert.momentum,
                'priority': alert.priority,
                'timestamp': alert.timestamp.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status in [200, 201, 204]:
                        print(f"✅ Webhook sent for {alert.ticker}")
                        return True
                    else:
                        print(f"❌ Webhook failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Webhook failed: {e}")
            return False
    
    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert through all configured channels"""
        # Check rate limits
        if not self.rate_limiter.can_send_alert(alert.ticker):
            print(f"⚠️ Rate limit exceeded for {alert.ticker}, alert skipped")
            return {}
        
        # Prioritize
        alert.priority = self.prioritize_alert(alert)
        
        # Send based on priority and channels
        results = {}
        
        # HIGH priority: All channels
        if alert.priority == 'HIGH':
            if 'email' in alert.channel:
                results['email'] = await self.send_email(alert)
            if 'discord' in alert.channel:
                results['discord'] = await self.send_discord(alert)
            if 'sms' in alert.channel:
                results['sms'] = await self.send_sms(alert)
        
        # MEDIUM priority: Email + Discord
        elif alert.priority == 'MEDIUM':
            if 'email' in alert.channel:
                results['email'] = await self.send_email(alert)
            if 'discord' in alert.channel:
                results['discord'] = await self.send_discord(alert)
        
        # LOW priority: Discord only
        else:
            if 'discord' in alert.channel:
                results['discord'] = await self.send_discord(alert)
        
        # Record alert
        self.rate_limiter.record_alert(alert.ticker)
        
        # Execute callbacks
        for callback in self.callbacks:
            try:
                await callback(alert)
            except Exception as e:
                print(f"❌ Callback failed: {e}")
        
        return results
    
    def register_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.callbacks.append(callback)
    
    async def process_scan_results(self, scan_results: Dict) -> List[Alert]:
        """Process watchlist scan results and generate alerts"""
        alerts = []
        
        for symbol, result in scan_results.items():
            # Generate alert for actionable signals
            if result.recommendation in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
                alert = Alert(
                    ticker=symbol,
                    signal=result.recommendation,
                    price=result.price,
                    signal_strength=result.signal_strength,
                    rsi_5m=result.rsi_5m,
                    rsi_1h=result.rsi_1h,
                    macd_signal=result.macd_signal,
                    momentum=result.momentum_score,
                    timestamp=result.timestamp,
                    priority='MEDIUM',  # Will be recalculated
                    channel=['email', 'discord']  # Default channels
                )
                
                # Send alert
                await self.send_alert(alert)
                alerts.append(alert)
        
        return alerts
    
    def export_alerts_log(self, filepath: str):
        """Export alert history to JSON"""
        data = {
            'total_alerts': len(self.alert_queue),
            'alerts_by_priority': {
                'HIGH': len([a for a in self.alert_queue if a.priority == 'HIGH']),
                'MEDIUM': len([a for a in self.alert_queue if a.priority == 'MEDIUM']),
                'LOW': len([a for a in self.alert_queue if a.priority == 'LOW'])
            },
            'alerts': [
                {
                    'ticker': a.ticker,
                    'signal': a.signal,
                    'price': a.price,
                    'signal_strength': a.signal_strength,
                    'priority': a.priority,
                    'timestamp': str(a.timestamp)
                }
                for a in self.alert_queue
            ]
        }
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Alert log exported to {filepath}")
