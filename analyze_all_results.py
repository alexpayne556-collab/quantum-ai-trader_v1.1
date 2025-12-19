#!/usr/bin/env python3
"""
ANALYZE ALL RESULTS - Quick Summary
Analyzes all comprehensive test results and shows top findings
"""

import pandas as pd
import glob
import os

def analyze_results():
    print("="*80)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    all_files = glob.glob('data/*COMPREHENSIVE.csv') + glob.glob('data/*_RESULTS.csv') + glob.glob('data/*_PHYSICS.csv')
    
    if not all_files:
        print("\n❌ No result files found yet!")
        print("   Run: python3 DEEP_FINANCIAL_PHYSICS.py all")
        return
    
    total_tested = 0
    total_significant = 0
    all_strategies = []
    
    for filepath in sorted(all_files):
        filename = os.path.basename(filepath)
        category = filename.replace('.csv', '').replace('_COMPREHENSIVE', '').replace('_RESULTS', '').replace('_PHYSICS', '')
        
        try:
            df = pd.read_csv(filepath)
            
            # Check if it has t_stat column
            if 't_stat' in df.columns:
                sig = df[df['t_stat'].abs() > 3.0]
                
                print(f"\n{'='*80}")
                print(f"📁 {category}")
                print(f"{'='*80}")
                print(f"   Total strategies: {len(df):,}")
                print(f"   Significant (t>3.0): {len(sig):,} ({len(sig)/len(df)*100:.1f}%)")
                
                if len(sig) > 0:
                    # Get top 5 by t-stat
                    top5 = sig.nlargest(5, 't_stat')
                    
                    print(f"\n   Top 5 Strategies:")
                    for idx, row in top5.iterrows():
                        strategy = row.get('strategy', f"Strategy_{idx}")
                        avg_net = row.get('avg_net', row.get('net_return', 0))
                        win_rate = row.get('win_rate', 0)
                        t_stat = row.get('t_stat', 0)
                        
                        print(f"      {strategy:30s}  Return: {avg_net:6.2%}  WR: {win_rate:5.1%}  t={t_stat:5.1f}")
                    
                    # Add to all strategies list
                    for idx, row in sig.iterrows():
                        all_strategies.append({
                            'category': category,
                            'strategy': row.get('strategy', f"Strategy_{idx}"),
                            'avg_net': row.get('avg_net', row.get('net_return', 0)),
                            'win_rate': row.get('win_rate', 0),
                            't_stat': row.get('t_stat', 0),
                            'n_tickers': row.get('n_tickers', row.get('num_signals', 0))
                        })
                
                total_tested += len(df)
                total_significant += len(sig)
            else:
                print(f"\n{'='*80}")
                print(f"📁 {category}")
                print(f"{'='*80}")
                print(f"   Rows: {len(df):,}")
        
        except Exception as e:
            print(f"\n❌ Error reading {filename}: {e}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("🎯 OVERALL SUMMARY")
    print("="*80)
    print(f"   Total strategies tested: {total_tested:,}")
    print(f"   Total significant (t>3.0): {total_significant:,}")
    if total_tested > 0:
        print(f"   Success rate: {total_significant/total_tested*100:.1f}%")
    
    # Top 20 across all categories
    if all_strategies:
        df_all = pd.DataFrame(all_strategies)
        top20 = df_all.nlargest(20, 't_stat')
        
        print(f"\n{'='*80}")
        print("🏆 TOP 20 STRATEGIES OVERALL (by t-statistic)")
        print("="*80)
        print(f"\n{'Category':<20} {'Strategy':<30} {'Return':>8} {'Win Rate':>9} {'T-Stat':>8}")
        print("-"*80)
        
        for idx, row in top20.iterrows():
            print(f"{row['category']:<20} {row['strategy']:<30} {row['avg_net']:>7.2%} {row['win_rate']:>8.1%} {row['t_stat']:>8.1f}")
        
        # Save top strategies
        top20.to_csv('data/TOP_20_STRATEGIES.csv', index=False)
        print(f"\n💾 Saved to: data/TOP_20_STRATEGIES.csv")
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review top strategies for deployment")
    print("  2. Run walk-forward validation on top 20")
    print("  3. Set up paper trading for top 5")
    print("")

if __name__ == "__main__":
    analyze_results()
