#!/usr/bin/env python3
"""
Prediction Market Ticker Generation Pipeline

Takes ~20K raw prediction market contracts, maps them to recurring Tickers,
and builds continuous time series by rolling contracts across expirations.

Usage: python run.py
Output: output/charts/*.png + output/results.xlsx
"""

import pandas as pd
import numpy as np
import json
import os
import re
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths
BASKET_ENGINE = Path("../basket-engine/data")
OUTPUT_DIR = Path("output")
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# STEP 1: Load and classify markets
# =============================================================================

def load_markets():
    """Load all markets and classifications."""
    print("=" * 60)
    print("STEP 1: Loading markets and classifications")
    print("=" * 60)
    
    markets = pd.read_parquet(BASKET_ENGINE / "processed/markets.parquet")
    print(f"  Total markets: {len(markets):,}")
    
    # Load classifications
    classifications = pd.read_parquet(BASKET_ENGINE / "processed/market_classifications.parquet")
    markets = markets.merge(classifications[['market_id', 'category']], on='market_id', how='left')
    
    # Filter stats
    category_counts = markets['category'].value_counts()
    sports = category_counts.get('sports', 0)
    entertainment = category_counts.get('entertainment', 0)
    print(f"  Sports markets: {sports:,}")
    print(f"  Entertainment markets: {entertainment:,}")
    
    filtered = markets[~markets['category'].isin(['sports', 'entertainment'])]
    print(f"  After filtering: {len(filtered):,}")
    
    return markets, filtered, category_counts


# =============================================================================
# STEP 2: Ticker mapping (CUSIP → Ticker)
# =============================================================================

def normalize_title(title):
    """Normalize market title to recurring Ticker concept."""
    if pd.isna(title):
        return ""
    
    n = str(title)
    
    # "after the [Month] [Year] meeting" → "after meeting"
    n = re.sub(r'\bafter\s+the\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+meeting\b',
               'after meeting', n, flags=re.I)
    n = re.sub(r'\bafter\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+meeting\b',
               'after meeting', n, flags=re.I)
    
    # "by [Month] [Day]" → "by [timeframe]"
    n = re.sub(r'\bby\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
               'by [timeframe]', n, flags=re.I)
    
    # Strip month-year combinations
    n = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', '', n, flags=re.I)
    
    # Strip standalone years
    n = re.sub(r'\b20[2-9]\d\b', '', n)
    
    # Strip standalone months
    n = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', '', n, flags=re.I)
    
    # Quarter references
    n = re.sub(r'\bQ[1-4]\b', '', n, flags=re.I)
    
    # Normalize "after meeting" variants
    n = re.sub(r'\bafter\s+(the\s+)?meeting\b', 'after meeting', n, flags=re.I)
    
    # "before [year]", "by [year]"
    n = re.sub(r'\bbefore\s+20[2-9]\d\b', 'before [year]', n, flags=re.I)
    n = re.sub(r'\bby\s+20[2-9]\d\b', 'by [year]', n, flags=re.I)
    
    # Date ranges
    n = re.sub(r'\b20[2-9]\d[-–]20[2-9]\d\b', '[daterange]', n)
    
    # Normalize dollar amounts: $150K → $150,000
    n = re.sub(r'\$(\d+)[kK]\b', lambda m: f'${m.group(1)},000', n)
    
    # Strip "Will the" / "Will" prefix (Polymarket changed format)
    n = re.sub(r'^Will\s+the\s+', '', n, flags=re.I)
    n = re.sub(r'^Will\s+', '', n, flags=re.I)
    
    # Normalize verb forms
    n = re.sub(r'\bdecreases\b', 'decrease', n, flags=re.I)
    n = re.sub(r'\bincreases\b', 'increase', n, flags=re.I)
    n = re.sub(r'\breaches\b', 'reach', n, flags=re.I)
    n = re.sub(r'\bwins\b', 'win', n, flags=re.I)
    
    # Clean trailing "in ?" artifact
    n = re.sub(r'\s+in\s*\?\s*$', '?', n)
    n = re.sub(r'\s+in\s*$', '', n)
    
    # Clean whitespace
    n = re.sub(r'\s+', ' ', n).strip()
    n = re.sub(r'^[^\w\$]+|[^\w\?\!]+$', '', n).strip()
    
    return n


def build_ticker_mapping(markets):
    """Map all CUSIPs to Tickers using exact matching on normalized titles."""
    print("\n" + "=" * 60)
    print("STEP 2: Building Ticker mapping")
    print("=" * 60)
    
    markets = markets.copy()
    markets['normalized'] = markets['title'].apply(normalize_title)
    markets['end_date_parsed'] = pd.to_datetime(markets['end_date'], errors='coerce')
    
    # Group by normalized title (exact match)
    ticker_groups = markets.groupby('normalized')
    
    ticker_mapping = []
    ticker_chains = {}
    ticker_id_counter = 0
    
    for norm_title, group in ticker_groups:
        if not norm_title:
            continue
        
        tid = f"ticker_{ticker_id_counter:06d}"
        ticker_id_counter += 1
        
        sorted_group = group.sort_values('end_date_parsed')
        
        for _, row in sorted_group.iterrows():
            ticker_mapping.append({
                'market_id': row['market_id'],
                'ticker_id': tid,
                'ticker_name': norm_title,
                'title': row['title'],
                'event_slug': row.get('event_slug', ''),
                'end_date': str(row.get('end_date', '')),
                'category': row.get('category', ''),
            })
        
        chain_markets = []
        for _, row in sorted_group.iterrows():
            chain_markets.append({
                'market_id': row['market_id'],
                'title': row['title'],
                'end_date': str(row.get('end_date', '')),
            })
        
        ticker_chains[tid] = {
            'ticker_id': tid,
            'ticker_name': norm_title,
            'market_count': len(group),
            'markets': chain_markets,
        }
    
    mapping_df = pd.DataFrame(ticker_mapping)
    
    total_tickers = len(ticker_chains)
    rollable = sum(1 for v in ticker_chains.values() if v['market_count'] >= 2)
    
    print(f"  Total Tickers: {total_tickers:,}")
    print(f"  Rollable (2+ CUSIPs): {rollable:,}")
    print(f"  Max CUSIPs per Ticker: {max(v['market_count'] for v in ticker_chains.values())}")
    
    # Distribution
    dist = Counter(v['market_count'] for v in ticker_chains.values())
    for k in sorted(dist.keys())[:8]:
        print(f"    {k} CUSIPs: {dist[k]:,} tickers")
    
    return mapping_df, ticker_chains


# =============================================================================
# STEP 3: Build continuous time series
# =============================================================================

def load_candle_data(market_id, markets_df):
    """Load price data for a single CUSIP."""
    row = markets_df[markets_df['market_id'] == market_id]
    if row.empty:
        return None
    row = row.iloc[0]
    
    platform = row.get('platform', '')
    
    if 'poly' in str(market_id).lower() or platform == 'polymarket':
        cond_id = row.get('condition_id', '')
        if not cond_id:
            return None
        candle_path = BASKET_ENGINE / f"raw/polymarket/candles_{cond_id}.json"
        if not candle_path.exists():
            return None
        try:
            with open(candle_path) as f:
                data = json.load(f)
            if not data:
                return None
            # Handle nested list format: data[0] is list of candle dicts
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                data = data[0]
            if not data or not isinstance(data, list):
                return None
            # Parse Polymarket candle format
            rows = []
            for item in data:
                ts = item.get('end_period_ts')
                price = item.get('price', {})
                close_str = price.get('close_dollars', None)
                vol = item.get('volume', 0)
                if ts and close_str:
                    rows.append({
                        'date': pd.to_datetime(ts, unit='s', errors='coerce'),
                        'close': float(close_str),
                        'volume': vol,
                    })
            if not rows:
                # Try flat format (t/c keys)
                df = pd.DataFrame(data)
                if 't' in df.columns:
                    df['date'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
                    df['close'] = pd.to_numeric(df.get('c', 0), errors='coerce')
                    df['volume'] = pd.to_numeric(df.get('v', 0), errors='coerce')
                else:
                    return None
            else:
                df = pd.DataFrame(rows)
            df = df.dropna(subset=['date', 'close'])
            df = df.sort_values('date').drop_duplicates('date')
            return df[['date', 'close', 'volume']].reset_index(drop=True)
        except:
            return None
    
    elif 'kalshi' in str(market_id).lower() or platform == 'kalshi':
        ticker = row.get('ticker', '')
        if not ticker:
            return None
        trade_path = BASKET_ENGINE / f"raw/kalshi/trades_{ticker}.json"
        if not trade_path.exists():
            return None
        try:
            with open(trade_path) as f:
                data = json.load(f)
            if not data:
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df.get('created_time', df.get('ts', '')), errors='coerce')
            df['close'] = pd.to_numeric(df.get('yes_price', df.get('price', 0)), errors='coerce') / 100.0
            df = df.dropna(subset=['date', 'close'])
            df['date'] = df['date'].dt.date
            daily = df.groupby('date').agg({'close': 'last'}).reset_index()
            daily['date'] = pd.to_datetime(daily['date'])
            daily['volume'] = 0
            return daily[['date', 'close', 'volume']].reset_index(drop=True)
        except:
            return None
    
    return None


def build_ticker_timeseries(ticker_chains, markets_df, min_days=30):
    """Build continuous time series for rollable Tickers."""
    print("\n" + "=" * 60)
    print("STEP 3: Building Ticker time series")
    print("=" * 60)
    
    rollable = {k: v for k, v in ticker_chains.items() if v['market_count'] >= 2}
    print(f"  Processing {len(rollable)} rollable Tickers...")
    
    all_raw = []
    all_adjusted = []
    stats = {}
    success = 0
    no_data = 0
    too_short = 0
    
    for i, (tid, chain) in enumerate(rollable.items()):
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(rollable)}")
        
        # Load candle data for each CUSIP in the chain
        cusip_data = []
        for m in chain['markets']:
            df = load_candle_data(m['market_id'], markets_df)
            if df is not None and len(df) > 0:
                end_date = pd.to_datetime(m['end_date'], errors='coerce')
                cusip_data.append({
                    'market_id': m['market_id'],
                    'title': m['title'],
                    'end_date': end_date,
                    'data': df,
                })
        
        if not cusip_data:
            no_data += 1
            continue
        
        # Sort by end_date
        cusip_data.sort(key=lambda x: x['end_date'] if pd.notna(x['end_date']) else pd.Timestamp.max)
        
        # Front-month rolling: use nearest expiry
        combined_rows = []
        used_dates = set()
        
        for ci, cusip in enumerate(cusip_data):
            df = cusip['data']
            is_last = (ci == len(cusip_data) - 1)
            
            if is_last:
                # Last CUSIP: use all remaining dates
                for _, row in df.iterrows():
                    d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                    if d not in used_dates:
                        combined_rows.append({
                            'date': row['date'],
                            'close': row['close'],
                            'volume': row.get('volume', 0),
                            'cusip': cusip['market_id'],
                            'is_roll': False,
                        })
                        used_dates.add(d)
            else:
                # Use until end_date or next CUSIP starts
                next_start = cusip_data[ci + 1]['data']['date'].min() if ci + 1 < len(cusip_data) else None
                end = cusip['end_date']
                
                for _, row in df.iterrows():
                    d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                    if d in used_dates:
                        continue
                    # Use this CUSIP's data up to its end date
                    if pd.notna(end) and row['date'] > end:
                        continue
                    combined_rows.append({
                        'date': row['date'],
                        'close': row['close'],
                        'volume': row.get('volume', 0),
                        'cusip': cusip['market_id'],
                        'is_roll': False,
                    })
                    used_dates.add(d)
        
        if len(combined_rows) < min_days:
            too_short += 1
            continue
        
        combined = pd.DataFrame(combined_rows).sort_values('date').reset_index(drop=True)
        
        # Mark roll points
        combined['is_roll'] = combined['cusip'] != combined['cusip'].shift(1)
        combined.iloc[0, combined.columns.get_loc('is_roll')] = False
        
        # Raw series
        combined['ticker_id'] = tid
        combined['ticker_name'] = chain['ticker_name']
        all_raw.append(combined)
        
        # Adjusted series (return-chained)
        adj = combined.copy()
        adj['daily_return'] = adj['close'].diff()
        # Zero out returns at roll points
        adj.loc[adj['is_roll'], 'daily_return'] = 0
        adj['daily_return'].iloc[0] = 0
        adj['adjusted_close'] = adj['close'].iloc[0] + adj['daily_return'].cumsum()
        all_adjusted.append(adj)
        
        # Stats
        roll_count = combined['is_roll'].sum()
        stats[tid] = {
            'ticker_name': chain['ticker_name'],
            'cusip_count': chain['market_count'],
            'data_points': len(combined),
            'duration_days': (combined['date'].max() - combined['date'].min()).days,
            'roll_count': int(roll_count),
            'start_date': str(combined['date'].min().date()),
            'end_date': str(combined['date'].max().date()),
            'category': chain.get('category', ''),
        }
        success += 1
    
    print(f"\n  Results:")
    print(f"    Successful: {success}")
    print(f"    No data: {no_data}")
    print(f"    Too short (<{min_days}d): {too_short}")
    
    raw_df = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    adj_df = pd.concat(all_adjusted, ignore_index=True) if all_adjusted else pd.DataFrame()
    
    print(f"    Total data points: {len(raw_df):,}")
    print(f"    Total roll points: {raw_df['is_roll'].sum() if len(raw_df) > 0 else 0}")
    
    return raw_df, adj_df, stats


# =============================================================================
# STEP 4: Generate charts
# =============================================================================

def chart_01_categories(category_counts):
    """Market distribution by category."""
    fig, ax = plt.subplots(figsize=(14, 7))
    top = category_counts.head(15)
    colors = ['#e74c3c' if c in ['sports', 'entertainment'] else '#3498db' for c in top.index]
    bars = ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel('Number of Markets')
    ax.set_title('Market Distribution by Category (red = filtered out)')
    ax.invert_yaxis()
    for i, v in enumerate(top.values):
        ax.text(v + 50, i, f'{v:,}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '01_category_distribution.png', dpi=150)
    plt.close()
    print("  ✓ 01_category_distribution.png")


def chart_02_ticker_distribution(ticker_chains):
    """CUSIPs per Ticker distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes = [v['market_count'] for v in ticker_chains.values()]
    
    # Histogram
    ax1.hist([s for s in sizes if s <= 20], bins=range(1, 22), color='#3498db', edgecolor='white')
    ax1.set_xlabel('CUSIPs per Ticker')
    ax1.set_ylabel('Number of Tickers')
    ax1.set_title('Distribution of CUSIPs per Ticker')
    
    # Top Tickers
    top = sorted(ticker_chains.values(), key=lambda x: -x['market_count'])[:15]
    names = [t['ticker_name'][:40] for t in top]
    counts = [t['market_count'] for t in top]
    ax2.barh(range(len(names)), counts, color='#2ecc71')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Number of CUSIPs')
    ax2.set_title('Top 15 Tickers by CUSIP Count')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '02_ticker_cusip_distribution.png', dpi=150)
    plt.close()
    print("  ✓ 02_ticker_cusip_distribution.png")


def chart_03_timeseries_samples(raw_df, stats, category_filter, filename, title):
    """Plot sample Ticker time series for a category."""
    if raw_df.empty:
        print(f"  ✗ {filename} (no data)")
        return
    
    # Get tickers for this category with most data
    cat_tickers = []
    for tid, s in stats.items():
        # Try to match category from ticker mapping
        ticker_data = raw_df[raw_df['ticker_id'] == tid]
        if ticker_data.empty:
            continue
        name = s['ticker_name'].lower()
        match = False
        for keyword in category_filter:
            if keyword in name:
                match = True
                break
        if match and s['data_points'] >= 60:
            cat_tickers.append((tid, s))
    
    cat_tickers.sort(key=lambda x: -x[1]['data_points'])
    cat_tickers = cat_tickers[:6]
    
    if not cat_tickers:
        print(f"  ✗ {filename} (no matching tickers)")
        return
    
    n = len(cat_tickers)
    fig, axes = plt.subplots(min(n, 3), max(1, (n + 2) // 3), figsize=(16, 4 * min(n, 3)), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, (tid, s) in enumerate(cat_tickers):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        data = raw_df[raw_df['ticker_id'] == tid].sort_values('date')
        ax.plot(data['date'], data['close'], linewidth=1.5, color='#2c3e50')
        
        # Mark roll points
        rolls = data[data['is_roll']]
        if len(rolls) > 0:
            ax.scatter(rolls['date'], rolls['close'], color='#e74c3c', s=30, zorder=5, label='Roll')
        
        ax.set_title(s['ticker_name'][:50], fontsize=10)
        ax.set_ylabel('Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        if len(rolls) > 0:
            ax.legend(fontsize=8)
    
    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  ✓ {filename}")


def chart_07_roll_analysis(raw_df, stats):
    """Roll points analysis."""
    if raw_df.empty:
        print("  ✗ 07_roll_points_analysis.png (no data)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Roll count distribution
    roll_counts = [s['roll_count'] for s in stats.values()]
    ax1.hist(roll_counts, bins=range(0, max(roll_counts) + 2), color='#9b59b6', edgecolor='white')
    ax1.set_xlabel('Number of Rolls')
    ax1.set_ylabel('Number of Tickers')
    ax1.set_title('Roll Count Distribution')
    
    # Duration vs rolls
    durations = [s['duration_days'] for s in stats.values()]
    ax2.scatter(durations, roll_counts, alpha=0.5, s=20, color='#e67e22')
    ax2.set_xlabel('Duration (days)')
    ax2.set_ylabel('Number of Rolls')
    ax2.set_title('Duration vs Roll Count')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '07_roll_points_analysis.png', dpi=150)
    plt.close()
    print("  ✓ 07_roll_points_analysis.png")


def chart_08_coverage(raw_df):
    """Data coverage timeline."""
    if raw_df.empty:
        print("  ✗ 08_coverage_timeline.png (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    daily = raw_df.groupby(raw_df['date'].dt.to_period('M')).agg(
        tickers=('ticker_id', 'nunique'),
        points=('close', 'count'),
    ).reset_index()
    daily['date'] = daily['date'].dt.to_timestamp()
    
    ax.bar(daily['date'], daily['tickers'], width=25, color='#3498db', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Active Tickers')
    ax.set_title('Active Ticker Coverage Over Time')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '08_coverage_timeline.png', dpi=150)
    plt.close()
    print("  ✓ 08_coverage_timeline.png")


def generate_all_charts(category_counts, ticker_chains, raw_df, stats):
    """Generate all charts."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating charts")
    print("=" * 60)
    
    chart_01_categories(category_counts)
    chart_02_ticker_distribution(ticker_chains)
    chart_03_timeseries_samples(raw_df, stats,
        ['fed', 'interest rate', 'monetary'],
        '03_sample_timeseries_fed.png', 'Fed Rate Decision Tickers')
    chart_03_timeseries_samples(raw_df, stats,
        ['bitcoin', 'ethereum', 'crypto', 'token'],
        '04_sample_timeseries_crypto.png', 'Crypto Tickers')
    chart_03_timeseries_samples(raw_df, stats,
        ['president', 'election', 'nominee', 'nomination'],
        '05_sample_timeseries_politics.png', 'Election Tickers')
    chart_03_timeseries_samples(raw_df, stats,
        ['iran', 'ukraine', 'china', 'taiwan', 'ceasefire', 'strike'],
        '06_sample_timeseries_geopolitics.png', 'Geopolitical Tickers')
    chart_07_roll_analysis(raw_df, stats)
    chart_08_coverage(raw_df)


# =============================================================================
# STEP 5: Write Excel output
# =============================================================================

def write_excel(markets, mapping_df, ticker_chains, stats, category_counts):
    """Write results to Excel."""
    print("\n" + "=" * 60)
    print("STEP 5: Writing results.xlsx")
    print("=" * 60)
    
    xlsx_path = OUTPUT_DIR / 'results.xlsx'
    
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Markets Ingested',
                'Sports Markets (filtered)',
                'Entertainment Markets (filtered)',
                'Markets After Filtering',
                'Unique Tickers',
                'Rollable Tickers (2+ CUSIPs)',
                'Tickers with Time Series',
                'Total Data Points',
                'Total Roll Points',
                'Date Range Start',
                'Date Range End',
            ],
            'Value': [
                len(markets),
                int(category_counts.get('sports', 0)),
                int(category_counts.get('entertainment', 0)),
                len(markets[~markets['category'].isin(['sports', 'entertainment'])]),
                len(ticker_chains),
                sum(1 for v in ticker_chains.values() if v['market_count'] >= 2),
                len(stats),
                sum(s['data_points'] for s in stats.values()),
                sum(s['roll_count'] for s in stats.values()),
                min((s['start_date'] for s in stats.values()), default='N/A'),
                max((s['end_date'] for s in stats.values()), default='N/A'),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Ticker Mapping sheet
        ticker_summary = []
        for tid, chain in ticker_chains.items():
            ticker_summary.append({
                'Ticker ID': tid,
                'Ticker Name': chain['ticker_name'],
                'CUSIP Count': chain['market_count'],
                'Rollable': 'Yes' if chain['market_count'] >= 2 else 'No',
            })
        pd.DataFrame(ticker_summary).sort_values('CUSIP Count', ascending=False).to_excel(
            writer, sheet_name='Ticker Mapping', index=False)
        
        # Ticker Chains sheet (rollable only)
        chain_rows = []
        for tid, chain in ticker_chains.items():
            if chain['market_count'] < 2:
                continue
            for i, m in enumerate(chain['markets']):
                chain_rows.append({
                    'Ticker ID': tid,
                    'Ticker Name': chain['ticker_name'],
                    'CUSIP #': i + 1,
                    'Market ID': m['market_id'],
                    'Title': m['title'],
                    'End Date': m['end_date'],
                })
        pd.DataFrame(chain_rows).to_excel(writer, sheet_name='Ticker Chains', index=False)
        
        # Time Series Stats sheet
        if stats:
            ts_rows = []
            for tid, s in stats.items():
                ts_rows.append({
                    'Ticker ID': tid,
                    'Ticker Name': s['ticker_name'],
                    'CUSIPs': s['cusip_count'],
                    'Data Points': s['data_points'],
                    'Duration (days)': s['duration_days'],
                    'Rolls': s['roll_count'],
                    'Start Date': s['start_date'],
                    'End Date': s['end_date'],
                })
            pd.DataFrame(ts_rows).sort_values('Data Points', ascending=False).to_excel(
                writer, sheet_name='Time Series Stats', index=False)
        
        # Market Classifications sheet
        class_df = markets[['market_id', 'title', 'category', 'event_slug', 'platform']].copy()
        class_df = class_df.sort_values('category')
        # Limit to 50K rows for Excel
        class_df.head(50000).to_excel(writer, sheet_name='Market Classifications', index=False)
    
    print(f"  ✓ {xlsx_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PREDICTION MARKET TICKER GENERATION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Step 1
    markets, filtered, category_counts = load_markets()
    
    # Step 2
    mapping_df, ticker_chains = build_ticker_mapping(markets)
    
    # Step 3
    raw_df, adj_df, stats = build_ticker_timeseries(ticker_chains, markets)
    
    # Step 4
    generate_all_charts(category_counts, ticker_chains, raw_df, stats)
    
    # Step 5
    write_excel(markets, mapping_df, ticker_chains, stats, category_counts)
    
    # Save processed data
    print("\n" + "=" * 60)
    print("STEP 6: Saving processed data")
    print("=" * 60)
    
    mapping_df.to_parquet(OUTPUT_DIR / 'ticker_mapping.parquet', index=False)
    print(f"  ✓ ticker_mapping.parquet")
    
    with open(OUTPUT_DIR / 'ticker_chains.json', 'w') as f:
        json.dump(ticker_chains, f, indent=2, default=str)
    print(f"  ✓ ticker_chains.json")
    
    if not raw_df.empty:
        raw_df.to_parquet(OUTPUT_DIR / 'ticker_timeseries_raw.parquet', index=False)
        adj_df.to_parquet(OUTPUT_DIR / 'ticker_timeseries_adjusted.parquet', index=False)
        print(f"  ✓ ticker_timeseries_raw.parquet")
        print(f"  ✓ ticker_timeseries_adjusted.parquet")
    
    with open(OUTPUT_DIR / 'ticker_timeseries_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ ticker_timeseries_stats.json")
    
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
