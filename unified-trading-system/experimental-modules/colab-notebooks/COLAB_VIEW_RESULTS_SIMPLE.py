"""
SIMPLE RESULTS VIEWER (No Cookie Issues)
=========================================

This version works even if third-party cookies are blocked.
It will show you everything in text format + save images to Drive.
"""

# ═══════════════════════════════════════════════════════════════════
# LOAD YOUR OVERNIGHT RESULTS
# ═══════════════════════════════════════════════════════════════════

from google.colab import drive
import sys, json, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no cookies needed)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from IPython.display import Image, display
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 LOADING YOUR TRAINING RESULTS")
print("="*80)

# Mount Drive
drive.mount('/content/drive', force_remount=False)
PROJECT_ROOT = "/content/drive/MyDrive/Quantum_AI_Cockpit"

print(f"✅ Drive: {PROJECT_ROOT}\n")

# Find all result files
report_files = sorted(pathlib.Path(PROJECT_ROOT).glob("MORNING_REPORT_*.json"))
diagnostic_files = sorted(pathlib.Path(PROJECT_ROOT).glob("diagnostic_*.json"))
progress_files = sorted(pathlib.Path(PROJECT_ROOT).glob("validation_progress_*.json"))

print("🔍 Found files:")
if report_files:
    for f in report_files:
        print(f"   📄 {f.name}")
if diagnostic_files:
    for f in diagnostic_files:
        print(f"   📄 {f.name}")
if progress_files:
    for f in progress_files:
        print(f"   📄 {f.name}")

# Load the most recent
data = None
source_file = None

if report_files:
    source_file = report_files[-1]
    with open(source_file) as f:
        data = json.load(f)
    print(f"\n✅ Loaded: {source_file.name}")
elif diagnostic_files:
    source_file = diagnostic_files[-1]
    with open(source_file) as f:
        data = json.load(f)
    print(f"\n✅ Loaded: {source_file.name}")
elif progress_files:
    source_file = progress_files[-1]
    with open(source_file) as f:
        data = json.load(f)
    print(f"\n✅ Loaded: {source_file.name}")
else:
    print("\n❌ No results found! Did the overnight training complete?")
    print("\nLooking for:")
    print("  - MORNING_REPORT_*.json")
    print("  - diagnostic_*.json")
    print("  - validation_progress_*.json")
    sys.exit()

# ═══════════════════════════════════════════════════════════════════
# ANALYZE RESULTS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 TRAINING RESULTS")
print("="*80)

results = data.get('results', [])
valid_results = [r for r in results if r.get('status') == 'ok']
failed_results = [r for r in results if r.get('status') not in ['ok', 'insufficient_data']]

print(f"\n✅ Total Stocks Tested: {len(results)}")
print(f"✅ Successful: {len(valid_results)}")
print(f"⚠️  Insufficient Data: {len([r for r in results if r.get('status') == 'insufficient_data'])}")
print(f"❌ Errors: {len(failed_results)}")

if failed_results:
    print("\n❌ Failed stocks:")
    for r in failed_results[:5]:  # Show first 5
        print(f"   {r['symbol']}: {r.get('error', 'Unknown')[:80]}")

if not valid_results:
    print("\n❌ No valid results to analyze!")
    sys.exit()

# Debug: Check what fields are available
print("\n🔍 Sample result fields:")
if valid_results:
    sample_keys = list(valid_results[0].keys())
    print(f"   Available fields: {', '.join(sample_keys)}")

# Create dataframe - handle different field name formats
df = pd.DataFrame([{
    'Symbol': r['symbol'],
    'Predictions': r.get('num_predictions', r.get('predictions', 0)),
    'Overall_Acc': r.get('overall_accuracy', r.get('accuracy', 0)),
    'BUY_Acc': r.get('buy_accuracy', 0),
    'BUY_Count': r.get('num_buys', 0),
    'SELL_Count': r.get('num_sells', 0),
    'HOLD_Count': r.get('num_holds', 0),
    'PNL': r.get('total_pnl_simulated', r.get('pnl', 0)),
    'Confidence': r.get('avg_confidence', r.get('confidence', 0)),
    'Issues': ', '.join(r.get('issues', []))
} for r in valid_results])

# ═══════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📈 PERFORMANCE SUMMARY")
print("="*80)

avg_overall_acc = df['Overall_Acc'].mean()
avg_buy_acc = df['BUY_Acc'].mean()
avg_confidence = df['Confidence'].mean()
total_pnl = df['PNL'].sum()
winners = len(df[df['PNL'] > 0])

print(f"\n🎯 Accuracy Metrics:")
print(f"   Overall Accuracy:     {avg_overall_acc:.1f}% (±{df['Overall_Acc'].std():.1f}%)")
print(f"   BUY Signal Accuracy:  {avg_buy_acc:.1f}% (±{df['BUY_Acc'].std():.1f}%)")
print(f"   Average Confidence:   {avg_confidence:.1f}%")

print(f"\n💰 Financial Performance:")
print(f"   Total Simulated P&L:  ${total_pnl:,.0f}")
print(f"   Winning Stocks:       {winners}/{len(df)} ({winners/len(df)*100:.1f}%)")
print(f"   Average P&L/Stock:    ${df['PNL'].mean():,.0f}")
print(f"   Best Stock P&L:       ${df['PNL'].max():,.0f}")
print(f"   Worst Stock P&L:      ${df['PNL'].min():,.0f}")

total_signals = df['BUY_Count'].sum() + df['SELL_Count'].sum() + df['HOLD_Count'].sum()
print(f"\n📡 Signal Distribution:")
print(f"   BUY:  {df['BUY_Count'].sum():4d}/{total_signals} ({df['BUY_Count'].sum()/total_signals*100:.1f}%)")
print(f"   SELL: {df['SELL_Count'].sum():4d}/{total_signals} ({df['SELL_Count'].sum()/total_signals*100:.1f}%)")
print(f"   HOLD: {df['HOLD_Count'].sum():4d}/{total_signals} ({df['HOLD_Count'].sum()/total_signals*100:.1f}%)")

# Calculate readiness score
score = data.get('readiness_score', 0)
if 'readiness_score' not in data:
    score = 0
    if avg_buy_acc >= 50:
        score += 50
    if total_pnl > 0:
        score += 30
    if df['SELL_Count'].sum() > 0:
        score += 20

print(f"\n{'='*80}")
print(f"🎯 READINESS SCORE: {score}/100")
print(f"{'='*80}")

if score >= 70:
    print("\n✅ SYSTEM READY FOR REAL MONEY TRADING!")
    print("   → Start with small positions ($500-1000)")
    print("   → Focus on top 5 performing stocks")
    print("   → Only take signals with >75% confidence")
    print("   → Use stop-loss at -5%")
elif score >= 50:
    print("\n⚠️  SYSTEM NEEDS IMPROVEMENT")
    print("   → Paper trade for 2 weeks first")
    print("   → Only trade top 3 stocks")
    print("   → Use $250 position sizes")
    print("   → Only take signals with >80% confidence")
else:
    print("\n❌ SYSTEM NOT READY FOR REAL MONEY")
    print("   → DO NOT trade yet")
    print("   → Major calibration needed")
    print("   → Review issues below")

# ═══════════════════════════════════════════════════════════════════
# TOP & BOTTOM PERFORMERS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🏆 TOP 5 PERFORMERS (by BUY accuracy)")
print("="*80)
top5 = df.nlargest(5, 'BUY_Acc')
for i, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"   {i}. {row['Symbol']:6s} → {row['BUY_Acc']:5.1f}% accuracy | ${row['PNL']:8,.0f} P&L | {row['BUY_Count']} BUY signals")

print("\n" + "="*80)
print("⚠️  BOTTOM 5 - AVOID THESE")
print("="*80)
bottom5 = df.nsmallest(5, 'BUY_Acc')
for i, (_, row) in enumerate(bottom5.iterrows(), 1):
    print(f"   {i}. {row['Symbol']:6s} → {row['BUY_Acc']:5.1f}% accuracy | ${row['PNL']:8,.0f} P&L | {row['BUY_Count']} BUY signals")

# ═══════════════════════════════════════════════════════════════════
# DETAILED STOCK BREAKDOWN
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📋 ALL STOCKS - DETAILED BREAKDOWN")
print("="*80)
print(f"\n{'Symbol':<8} {'BUY%':>6} {'P&L':>10} {'BUY':>5} {'SELL':>5} {'HOLD':>5} {'Issues':<30}")
print("-"*80)

for _, row in df.sort_values('BUY_Acc', ascending=False).iterrows():
    issues_str = row['Issues'][:28] if row['Issues'] else 'None'
    print(f"{row['Symbol']:<8} {row['BUY_Acc']:>6.1f} ${row['PNL']:>9,.0f} "
          f"{row['BUY_Count']:>5} {row['SELL_Count']:>5} {row['HOLD_Count']:>5} {issues_str:<30}")

# ═══════════════════════════════════════════════════════════════════
# CREATE CHARTS (Saved to Drive)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)
print("\nSaving charts to Google Drive (no cookie issues!)...\n")

sns.set_style("whitegrid")

# Chart 1: BUY Accuracy
fig1, ax1 = plt.subplots(figsize=(12, 8))
df_sorted = df.sort_values('BUY_Acc', ascending=True)
colors = ['#2ecc71' if x >= 50 else '#e74c3c' for x in df_sorted['BUY_Acc']]
bars = ax1.barh(df_sorted['Symbol'], df_sorted['BUY_Acc'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.axvline(50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax1.set_xlabel('BUY Signal Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title(f'📈 BUY Signal Accuracy by Stock\n(Readiness Score: {score}/100)', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(axis='x', alpha=0.4)
for bar in bars:
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
chart1_path = pathlib.Path(PROJECT_ROOT) / "chart_buy_accuracy.png"
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: chart_buy_accuracy.png")
plt.close()

# Chart 2: P&L
fig2, ax2 = plt.subplots(figsize=(12, 8))
df_pnl = df.sort_values('PNL', ascending=True)
colors2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_pnl['PNL']]
bars2 = ax2.barh(df_pnl['Symbol'], df_pnl['PNL'], color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
ax2.axvline(0, color='black', linestyle='-', linewidth=2.5)
ax2.set_xlabel('Simulated P&L ($10K position per trade)', fontsize=14, fontweight='bold')
ax2.set_title(f'💰 Profitability by Stock\n(Total: ${total_pnl:,.0f})', fontsize=16, fontweight='bold')
ax2.grid(axis='x', alpha=0.4)
for bar in bars2:
    width = bar.get_width()
    label_x = width + 50 if width > 0 else width - 50
    ax2.text(label_x, bar.get_y() + bar.get_height()/2, f'${width:,.0f}',
            ha='left' if width > 0 else 'right', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
chart2_path = pathlib.Path(PROJECT_ROOT) / "chart_pnl.png"
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: chart_pnl.png")
plt.close()

# Chart 3: Signal Distribution
fig3, ax3 = plt.subplots(figsize=(10, 8))
signal_totals = [df['BUY_Count'].sum(), df['SELL_Count'].sum(), df['HOLD_Count'].sum()]
signal_labels = [f'BUY\n({signal_totals[0]})', f'SELL\n({signal_totals[1]})', f'HOLD\n({signal_totals[2]})']
colors3 = ['#2ecc71', '#e74c3c', '#95a5a6']
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = ax3.pie(signal_totals, labels=signal_labels, colors=colors3,
                                     autopct='%1.1f%%', startangle=90, explode=explode,
                                     textprops={'fontsize': 14, 'fontweight': 'bold'},
                                     shadow=True)
ax3.set_title('📊 Signal Distribution', fontsize=18, fontweight='bold')
plt.tight_layout()
chart3_path = pathlib.Path(PROJECT_ROOT) / "chart_signals.png"
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: chart_signals.png")
plt.close()

# Chart 4: Accuracy Scatter
fig4, ax4 = plt.subplots(figsize=(12, 8))
scatter = ax4.scatter(df['Overall_Acc'], df['BUY_Acc'],
                     s=np.abs(df['PNL'])/3,
                     c=df['PNL'], cmap='RdYlGn',
                     alpha=0.7, edgecolors='black', linewidth=2)
ax4.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax4.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax4.set_xlabel('Overall Accuracy (%)', fontsize=14, fontweight='bold')
ax4.set_ylabel('BUY Accuracy (%)', fontsize=14, fontweight='bold')
ax4.set_title('🎯 Accuracy Matrix (bubble size = P&L)', fontsize=16, fontweight='bold')
ax4.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='P&L ($)')
for _, row in df.iterrows():
    ax4.annotate(row['Symbol'], (row['Overall_Acc'], row['BUY_Acc']),
                fontsize=10, fontweight='bold', alpha=0.8)
plt.tight_layout()
chart4_path = pathlib.Path(PROJECT_ROOT) / "chart_accuracy_matrix.png"
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: chart_accuracy_matrix.png")
plt.close()

print(f"\n✅ All charts saved to: {PROJECT_ROOT}")

# ═══════════════════════════════════════════════════════════════════
# DISPLAY CHARTS IN COLAB
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 DISPLAYING CHARTS")
print("="*80 + "\n")

try:
    display(Image(filename=str(chart1_path)))
    print("\n")
    display(Image(filename=str(chart2_path)))
    print("\n")
    display(Image(filename=str(chart3_path)))
    print("\n")
    display(Image(filename=str(chart4_path)))
except Exception as e:
    print(f"⚠️  Could not display images inline: {e}")
    print(f"\n📁 But they are saved to your Drive! Open these files:")
    print(f"   - chart_buy_accuracy.png")
    print(f"   - chart_pnl.png")
    print(f"   - chart_signals.png")
    print(f"   - chart_accuracy_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# ACTIONABLE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("💡 WHAT TO DO NEXT")
print("="*80)

if score >= 70:
    print("\n✅ YOUR SYSTEM IS READY!")
    print("\n📋 Trading Plan:")
    print(f"   1. Focus on these stocks: {', '.join(top5['Symbol'].tolist())}")
    print(f"   2. Position size: $500-1000 per trade")
    print(f"   3. Only take signals with >75% confidence")
    print(f"   4. Use stop-loss at -5%")
    print(f"   5. Start with paper trading for 1 week to verify")
    
    print("\n🚀 Next Steps:")
    print("   → Build the trading UI dashboard")
    print("   → Set up real-time price monitoring")
    print("   → Create trade execution interface")
    print("   → Implement portfolio tracking")

elif score >= 50:
    print("\n⚠️  SYSTEM NEEDS TUNING")
    print("\n📋 Improvement Plan:")
    issues = []
    if df['SELL_Count'].sum() == 0:
        issues.append("Add SELL signal logic")
    if df['BUY_Count'].sum() / total_signals > 0.85:
        issues.append("Reduce bullish bias")
    if avg_confidence > avg_overall_acc + 10:
        issues.append("Fix overconfidence")
    
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n🚀 Next Steps:")
    print("   → Fix identified issues")
    print("   → Re-run overnight training")
    print("   → Paper trade top 3 stocks for 2 weeks")

else:
    print("\n❌ SYSTEM NOT READY")
    print("\n📋 Critical Issues to Fix:")
    print(f"   - BUY accuracy too low ({avg_buy_acc:.1f}% < 50%)")
    if df['SELL_Count'].sum() == 0:
        print(f"   - No SELL signals (always bullish)")
    if total_pnl < 0:
        print(f"   - Unprofitable (${total_pnl:,.0f})")
    
    print("\n🚀 Next Steps:")
    print("   → Review module configurations")
    print("   → Check data quality")
    print("   → Test individual modules")
    print("   → Re-run training after fixes")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 All files in: {PROJECT_ROOT}")
print(f"📊 Charts: chart_*.png")
print(f"📄 Raw data: {source_file.name}")
print("\n💡 Tip: Download the charts from Google Drive to share/review later!\n")

