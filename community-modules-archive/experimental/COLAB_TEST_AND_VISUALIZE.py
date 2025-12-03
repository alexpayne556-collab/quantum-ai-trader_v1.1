"""
COMPREHENSIVE TESTING & VISUALIZATION SUITE
============================================

This cell will:
1. Test that training worked
2. Evaluate each module individually
3. Create beautiful visualizations
4. Show you what's working and what needs fixing

Run this after overnight training completes.
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1: SETUP & LOAD RESULTS
# ═══════════════════════════════════════════════════════════════════

from google.colab import drive
import sys, json, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Mount and load
print("📁 Mounting Drive...")
drive.mount('/content/drive', force_remount=False)
PROJECT_ROOT = "/content/drive/MyDrive/Quantum_AI_Cockpit"

print(f"✅ {PROJECT_ROOT}")
print("\n🔍 Looking for training results...\n")

# Find latest morning report
report_files = sorted(pathlib.Path(PROJECT_ROOT).glob("MORNING_REPORT_*.json"))
diagnostic_files = sorted(pathlib.Path(PROJECT_ROOT).glob("diagnostic_*.json"))
progress_files = sorted(pathlib.Path(PROJECT_ROOT).glob("validation_progress_*.json"))

if not report_files and not diagnostic_files and not progress_files:
    print("❌ No training results found!")
    print("\nMake sure you ran the overnight training cell first.")
    print("Looking for files matching:")
    print("  - MORNING_REPORT_*.json")
    print("  - diagnostic_*.json")
    print("  - validation_progress_*.json")
else:
    print("✅ Found training results:")
    if report_files:
        print(f"   📄 {report_files[-1].name}")
    if diagnostic_files:
        print(f"   📄 {diagnostic_files[-1].name}")
    if progress_files:
        print(f"   📄 {progress_files[-1].name}")

# Load the most recent results
data = None
if report_files:
    with open(report_files[-1]) as f:
        data = json.load(f)
    print(f"\n📊 Loaded: {report_files[-1].name}")
elif diagnostic_files:
    with open(diagnostic_files[-1]) as f:
        data = json.load(f)
    print(f"\n📊 Loaded: {diagnostic_files[-1].name}")
elif progress_files:
    with open(progress_files[-1]) as f:
        data = json.load(f)
    print(f"\n📊 Loaded: {progress_files[-1].name}")

if data is None:
    print("\n❌ Could not load any results!")
else:
    print("✅ Results loaded successfully!")

# ═══════════════════════════════════════════════════════════════════
# CELL 2: ANALYZE RESULTS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 TRAINING RESULTS ANALYSIS")
print("="*80)

results = data.get('results', [])
valid_results = [r for r in results if r.get('status') == 'ok']
failed_results = [r for r in results if r.get('status') not in ['ok', 'insufficient_data']]

print(f"\nTotal Stocks: {len(results)}")
print(f"✅ Successful: {len(valid_results)}")
print(f"⚠️  Insufficient Data: {len([r for r in results if r.get('status') == 'insufficient_data'])}")
print(f"❌ Errors: {len(failed_results)}")

if failed_results:
    print("\n⚠️  Failed stocks:")
    for r in failed_results:
        print(f"   {r['symbol']}: {r.get('error', 'Unknown error')}")

if not valid_results:
    print("\n❌ No valid results to analyze!")
else:
    # Create summary dataframe
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Predictions': r['num_predictions'],
        'Overall_Acc': r['overall_accuracy'],
        'BUY_Acc': r['buy_accuracy'],
        'BUY_Count': r['num_buys'],
        'SELL_Count': r['num_sells'],
        'HOLD_Count': r['num_holds'],
        'PNL': r['total_pnl_simulated'],
        'Confidence': r['avg_confidence'],
        'Issues': ', '.join(r.get('issues', []))
    } for r in valid_results])
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nOverall Accuracy:     {df['Overall_Acc'].mean():.1f}% (±{df['Overall_Acc'].std():.1f}%)")
    print(f"BUY Signal Accuracy:  {df['BUY_Acc'].mean():.1f}% (±{df['BUY_Acc'].std():.1f}%)")
    print(f"Average Confidence:   {df['Confidence'].mean():.1f}%")
    print(f"\nTotal Simulated P&L:  ${df['PNL'].sum():,.0f}")
    print(f"Winning Stocks:       {len(df[df['PNL'] > 0])}/{len(df)} ({len(df[df['PNL'] > 0])/len(df)*100:.1f}%)")
    print(f"Average P&L/Stock:    ${df['PNL'].mean():,.0f}")
    
    # Signal distribution
    total_signals = df['BUY_Count'].sum() + df['SELL_Count'].sum() + df['HOLD_Count'].sum()
    print(f"\n📊 Signal Distribution:")
    print(f"   BUY:  {df['BUY_Count'].sum()}/{total_signals} ({df['BUY_Count'].sum()/total_signals*100:.1f}%)")
    print(f"   SELL: {df['SELL_Count'].sum()}/{total_signals} ({df['SELL_Count'].sum()/total_signals*100:.1f}%)")
    print(f"   HOLD: {df['HOLD_Count'].sum()}/{total_signals} ({df['HOLD_Count'].sum()/total_signals*100:.1f}%)")
    
    # Issues analysis
    all_issues = []
    for r in valid_results:
        all_issues.extend(r.get('issues', []))
    
    if all_issues:
        issue_counts = pd.Series(all_issues).value_counts()
        print(f"\n⚠️  Common Issues:")
        for issue, count in issue_counts.items():
            print(f"   {issue}: {count} stocks")
    else:
        print("\n✅ No major issues detected!")
    
    # Readiness score
    score = data.get('readiness_score', 0)
    if 'readiness_score' not in data:
        # Calculate it
        score = 0
        if df['BUY_Acc'].mean() >= 50:
            score += 50
        if df['PNL'].sum() > 0:
            score += 30
        if df['SELL_Count'].sum() > 0:
            score += 20
    
    print(f"\n{'='*80}")
    print(f"🎯 READINESS SCORE: {score}/100")
    print(f"{'='*80}")
    
    if score >= 70:
        print("\n✅ SYSTEM READY FOR REAL MONEY")
        print("   → Start with small position sizes ($500-1000)")
        print("   → Focus on top 5 performing stocks")
        print("   → Use stop losses at -5%")
    elif score >= 50:
        print("\n⚠️  SYSTEM NEEDS IMPROVEMENT")
        print("   → Paper trade for 2 weeks first")
        print("   → Fix identified issues before going live")
        print("   → Consider only trading top 3 stocks")
    else:
        print("\n❌ SYSTEM NOT READY")
        print("   → DO NOT trade real money")
        print("   → Major calibration needed")
        print("   → Review module performance below")

# ═══════════════════════════════════════════════════════════════════
# CELL 3: BEAUTIFUL VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

if valid_results:
    print("\n\n" + "="*80)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('🎯 Quantum AI Cockpit - Complete Training Analysis', 
                 fontsize=24, fontweight='bold', y=0.995)
    
    # ==================== ROW 1 ====================
    
    # 1. BUY Accuracy by Stock (sorted)
    ax1 = fig.add_subplot(gs[0, :2])
    df_sorted = df.sort_values('BUY_Acc', ascending=True)
    colors1 = ['#2ecc71' if x >= 50 else '#e74c3c' for x in df_sorted['BUY_Acc']]
    bars1 = ax1.barh(df_sorted['Symbol'], df_sorted['BUY_Acc'], color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axvline(50, color='black', linestyle='--', linewidth=2, alpha=0.6, label='50% Threshold')
    ax1.set_xlabel('BUY Signal Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('📈 BUY Signal Accuracy by Stock', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='x', alpha=0.4, linestyle='--')
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 2. Summary Stats Box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = f"""
╔═══════════════════════════════════╗
║   SYSTEM PERFORMANCE SUMMARY      ║
╚═══════════════════════════════════╝

📊 Stocks Tested: {len(df)}/{len(results)}

🎯 Accuracy Metrics:
   Overall:      {df['Overall_Acc'].mean():.1f}%
   BUY Signals:  {df['BUY_Acc'].mean():.1f}%
   Avg Confidence: {df['Confidence'].mean():.1f}%

💰 Financial Performance:
   Total P&L:    ${df['PNL'].sum():,.0f}
   Winners:      {len(df[df['PNL'] > 0])}/{len(df)}
   Win Rate:     {len(df[df['PNL'] > 0])/len(df)*100:.1f}%

📡 Signal Distribution:
   BUY:  {df['BUY_Count'].sum()}/{total_signals} ({df['BUY_Count'].sum()/total_signals*100:.0f}%)
   SELL: {df['SELL_Count'].sum()}/{total_signals} ({df['SELL_Count'].sum()/total_signals*100:.0f}%)
   HOLD: {df['HOLD_Count'].sum()}/{total_signals} ({df['HOLD_Count'].sum()/total_signals*100:.0f}%)

🎯 READINESS: {score}/100 {'✅' if score >= 70 else '⚠️' if score >= 50 else '❌'}
    """
    
    ax2.text(0.05, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=2))
    
    # ==================== ROW 2 ====================
    
    # 3. P&L by Stock (sorted)
    ax3 = fig.add_subplot(gs[1, :2])
    df_pnl = df.sort_values('PNL', ascending=True)
    colors3 = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_pnl['PNL']]
    bars3 = ax3.barh(df_pnl['Symbol'], df_pnl['PNL'], color=colors3, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=2.5)
    ax3.set_xlabel('Simulated P&L ($10K position)', fontsize=13, fontweight='bold')
    ax3.set_title('💰 Profitability by Stock', fontsize=16, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.4, linestyle='--')
    # Add value labels
    for bar in bars3:
        width = bar.get_width()
        label_x = width + 50 if width > 0 else width - 50
        ax3.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'${width:,.0f}', ha='left' if width > 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')
    
    # 4. Signal Distribution Pie Chart
    ax4 = fig.add_subplot(gs[1, 2])
    signal_totals = [df['BUY_Count'].sum(), df['SELL_Count'].sum(), df['HOLD_Count'].sum()]
    signal_labels = ['BUY', 'SELL', 'HOLD']
    colors4 = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax4.pie(signal_totals, labels=signal_labels, colors=colors4,
                                         autopct='%1.1f%%', startangle=90, explode=explode,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'},
                                         shadow=True)
    ax4.set_title('📊 Signal Distribution', fontsize=16, fontweight='bold', pad=15)
    
    # ==================== ROW 3 ====================
    
    # 5. Accuracy Scatter Plot
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(df['Overall_Acc'], df['BUY_Acc'], 
                         s=np.abs(df['PNL'])/5, 
                         c=df['PNL'], cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black', linewidth=2)
    ax5.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax5.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax5.set_xlabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('BUY Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_title('🎯 Accuracy Matrix\n(size=P&L)', fontsize=14, fontweight='bold')
    ax5.grid(alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax5, label='P&L ($)', shrink=0.8)
    # Add labels for outliers
    for _, row in df.iterrows():
        if row['BUY_Acc'] > 60 or row['BUY_Acc'] < 40 or abs(row['PNL']) > 500:
            ax5.annotate(row['Symbol'], (row['Overall_Acc'], row['BUY_Acc']), 
                        fontsize=8, fontweight='bold', alpha=0.8)
    
    # 6. Confidence vs Accuracy
    ax6 = fig.add_subplot(gs[2, 1])
    colors6 = ['#2ecc71' if x >= y else '#e74c3c' for x, y in zip(df['Overall_Acc'], df['Confidence'])]
    ax6.scatter(df['Confidence'], df['Overall_Acc'], s=200, c=colors6, alpha=0.7, edgecolors='black', linewidth=2)
    # Add diagonal line (perfect calibration)
    lims = [max(ax6.get_xlim()[0], ax6.get_ylim()[0]), min(ax6.get_xlim()[1], ax6.get_ylim()[1])]
    ax6.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Calibration')
    ax6.set_xlabel('Average Confidence (%)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Actual Accuracy (%)', fontsize=12, fontweight='bold')
    ax6.set_title('🎲 Confidence Calibration\n(Green=Good, Red=Overconfident)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--')
    # Add labels
    for _, row in df.iterrows():
        ax6.annotate(row['Symbol'], (row['Confidence'], row['Overall_Acc']), 
                    fontsize=7, alpha=0.7)
    
    # 7. Prediction Volume
    ax7 = fig.add_subplot(gs[2, 2])
    df_pred = df.sort_values('Predictions', ascending=True)
    bars7 = ax7.barh(df_pred['Symbol'], df_pred['Predictions'], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax7.set_xlabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax7.set_title('📈 Prediction Volume', fontsize=14, fontweight='bold')
    ax7.grid(axis='x', alpha=0.4, linestyle='--')
    
    # ==================== ROW 4 ====================
    
    # 8. Top Performers
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.axis('off')
    
    top5 = df.nlargest(5, 'BUY_Acc')
    top_text = "🏆 TOP 5 PERFORMERS\n" + "="*35 + "\n\n"
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        top_text += f"{i}. {row['Symbol']:6s} → {row['BUY_Acc']:5.1f}% | ${row['PNL']:7,.0f}\n"
    
    ax8.text(0.1, 0.5, top_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='#d5f4e6', alpha=0.8, edgecolor='green', linewidth=2))
    
    # 9. Bottom Performers
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.axis('off')
    
    bottom5 = df.nsmallest(5, 'BUY_Acc')
    bottom_text = "⚠️  BOTTOM 5 - AVOID\n" + "="*35 + "\n\n"
    for i, (_, row) in enumerate(bottom5.iterrows(), 1):
        bottom_text += f"{i}. {row['Symbol']:6s} → {row['BUY_Acc']:5.1f}% | ${row['PNL']:7,.0f}\n"
    
    ax9.text(0.1, 0.5, bottom_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='#fde2e4', alpha=0.8, edgecolor='red', linewidth=2))
    
    # 10. Issues Heatmap
    ax10 = fig.add_subplot(gs[3, 2])
    if all_issues:
        issue_counts = pd.Series(all_issues).value_counts()
        colors10 = plt.cm.Reds(np.linspace(0.3, 0.9, len(issue_counts)))
        bars10 = ax10.barh(range(len(issue_counts)), issue_counts.values, color=colors10, 
                          edgecolor='black', linewidth=1.5)
        ax10.set_yticks(range(len(issue_counts)))
        ax10.set_yticklabels(issue_counts.index, fontsize=10)
        ax10.set_xlabel('Number of Stocks', fontsize=12, fontweight='bold')
        ax10.set_title('⚠️  Issue Frequency', fontsize=14, fontweight='bold')
        ax10.grid(axis='x', alpha=0.4, linestyle='--')
        # Add count labels
        for i, bar in enumerate(bars10):
            ax10.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{int(bar.get_width())}', ha='left', va='center', 
                     fontsize=10, fontweight='bold')
    else:
        ax10.text(0.5, 0.5, '✅ NO ISSUES!', ha='center', va='center', 
                 fontsize=20, fontweight='bold', color='green',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.5))
        ax10.set_xticks([])
        ax10.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Visualizations complete!")

# ═══════════════════════════════════════════════════════════════════
# CELL 4: MODULE-BY-MODULE TESTING
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("🔬 MODULE-BY-MODULE PERFORMANCE ANALYSIS")
print("="*80)

if valid_results and len(valid_results) > 0:
    print("\nAnalyzing individual predictions to identify module strengths/weaknesses...")
    
    # Aggregate all predictions
    all_predictions = []
    for result in valid_results:
        for pred in result.get('predictions', []):
            pred['symbol'] = result['symbol']
            all_predictions.append(pred)
    
    pred_df = pd.DataFrame(all_predictions)
    
    if len(pred_df) > 0:
        print(f"\nTotal predictions analyzed: {len(pred_df)}")
        
        # Accuracy by action type
        print("\n📊 Accuracy by Signal Type:")
        for action in ['BUY', 'STRONG_BUY', 'BUY_THE_DIP', 'SELL', 'HOLD']:
            action_df = pred_df[pred_df['action'] == action]
            if len(action_df) > 0:
                accuracy = (action_df['correct'].sum() / len(action_df)) * 100
                avg_return = action_df['actual_return_pct'].mean()
                print(f"   {action:12s}: {accuracy:5.1f}% accuracy | {len(action_df):4d} signals | Avg return: {avg_return:+6.2f}%")
        
        # Confidence bucket analysis
        print("\n🎲 Performance by Confidence Level:")
        pred_df['conf_bucket'] = pd.cut(pred_df['confidence'], bins=[0, 0.6, 0.75, 0.85, 1.0], 
                                         labels=['Low (0-60%)', 'Medium (60-75%)', 'High (75-85%)', 'Very High (85%+)'])
        for bucket in pred_df['conf_bucket'].unique():
            if pd.notna(bucket):
                bucket_df = pred_df[pred_df['conf_bucket'] == bucket]
                accuracy = (bucket_df['correct'].sum() / len(bucket_df)) * 100
                print(f"   {bucket:20s}: {accuracy:5.1f}% accuracy | {len(bucket_df):4d} predictions")
        
        # Best/worst prediction days
        print("\n📅 Temporal Analysis:")
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        daily_acc = pred_df.groupby(pred_df['date'].dt.date).agg({
            'correct': 'mean',
            'actual_return_pct': 'mean',
            'symbol': 'count'
        }).rename(columns={'symbol': 'count', 'correct': 'accuracy'})
        daily_acc['accuracy'] *= 100
        
        best_day = daily_acc.nlargest(1, 'accuracy')
        worst_day = daily_acc.nsmallest(1, 'accuracy')
        
        print(f"   Best day:  {best_day.index[0]} - {best_day['accuracy'].iloc[0]:.1f}% accuracy ({int(best_day['count'].iloc[0])} predictions)")
        print(f"   Worst day: {worst_day.index[0]} - {worst_day['accuracy'].iloc[0]:.1f}% accuracy ({int(worst_day['count'].iloc[0])} predictions)")
        
        # Volatility analysis
        print("\n📈 Performance by Market Condition:")
        pred_df['volatility'] = pd.cut(pred_df['actual_return_pct'].abs(), 
                                       bins=[0, 2, 5, 100],
                                       labels=['Low Volatility (<2%)', 'Medium Volatility (2-5%)', 'High Volatility (>5%)'])
        for vol in pred_df['volatility'].unique():
            if pd.notna(vol):
                vol_df = pred_df[pred_df['volatility'] == vol]
                accuracy = (vol_df['correct'].sum() / len(vol_df)) * 100
                print(f"   {vol:30s}: {accuracy:5.1f}% accuracy | {len(vol_df):4d} predictions")
        
        print("\n✅ Module analysis complete!")
    else:
        print("⚠️  No individual predictions found in results")

# ═══════════════════════════════════════════════════════════════════
# CELL 5: ACTIONABLE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("💡 ACTIONABLE RECOMMENDATIONS")
print("="*80)

if valid_results:
    print("\nBased on the training results, here's what you should do:\n")
    
    # Trading recommendations
    print("🎯 TRADING STRATEGY:")
    if score >= 70:
        print("   ✅ You can start trading with real money")
        print(f"   → Focus on these stocks: {', '.join(df.nlargest(5, 'BUY_Acc')['Symbol'].tolist())}")
        print("   → Position size: $500-1000 per trade")
        print("   → Only take BUY signals with >75% confidence")
        print("   → Use stop-loss at -5%")
    elif score >= 50:
        print("   ⚠️  Paper trade for 2 weeks first")
        print(f"   → Only trade top 3 stocks: {', '.join(df.nlargest(3, 'BUY_Acc')['Symbol'].tolist())}")
        print("   → When live, start with $250 per trade")
        print("   → Only take signals with >80% confidence")
    else:
        print("   ❌ DO NOT trade real money yet")
        print("   → System needs significant improvement")
        print("   → Review issues identified above")
    
    # System improvements
    print("\n🔧 SYSTEM IMPROVEMENTS NEEDED:")
    
    if df['SELL_Count'].sum() == 0:
        print("   ⚠️  CRITICAL: System never issues SELL signals")
        print("       → Add bearish pattern detection")
        print("       → Lower SELL confidence threshold")
    
    if df['BUY_Count'].sum() / total_signals > 0.85:
        print("   ⚠️  System is too bullish-biased")
        print("       → Increase confidence threshold for BUY signals")
        print("       → Enable more neutral/bearish modules")
    
    if df['Confidence'].mean() > df['Overall_Acc'].mean() + 10:
        print("   ⚠️  System is overconfident")
        print("       → Reduce confidence calculation multipliers")
        print("       → Add uncertainty penalties")
    
    # Module specific
    if 'OVERCONFIDENT' in all_issues:
        print("   ⚠️  Confidence calibration needed")
        print("       → Adjust fusior_forecast.py confidence calculations")
    
    if 'LOW_BUY_ACCURACY' in all_issues:
        print("   ⚠️  BUY signals are unreliable")
        print("       → Review pattern_integration_layer.py")
        print("       → Check EMA and trend detection modules")
    
    # Data quality
    if failed_results:
        print(f"\n📊 DATA QUALITY:")
        print(f"   ⚠️  {len(failed_results)} stocks had errors")
        print("       → Check API keys and rate limits")
        print("       → Verify data_orchestrator fallback logic")
    
    print("\n✅ Recommendations complete!")
    
    # Next steps
    print("\n" + "="*80)
    print("🚀 NEXT STEPS")
    print("="*80)
    print("\n1. Review the visualizations above")
    print("2. Address critical issues identified")
    print("3. If score ≥70: Start building the trading UI")
    print("4. If score <70: Re-run training after fixes")
    print("\n📁 All results saved to Google Drive")
    print(f"   Location: {PROJECT_ROOT}")

print("\n\n" + "="*80)
print("✅ COMPLETE TESTING & VISUALIZATION FINISHED!")
print("="*80)

