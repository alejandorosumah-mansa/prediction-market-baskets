#!/usr/bin/env python3
"""
Prediction Market Ticker Generation Pipeline

Takes ~20K raw prediction market contracts, maps them to recurring Tickers,
and builds continuous time series by rolling contracts across expirations.

Usage: python run.py
Output: output/charts/*.png (one per Ticker) + output/results.xlsx (one sheet per Ticker)
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
# STEP 4: Generate individual ticker charts
# =============================================================================

def sanitize_filename(name):
    """Sanitize ticker name for use as filename."""
    # Remove or replace characters that aren't safe for filenames
    name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Windows-unsafe chars
    name = re.sub(r'[^\w\s\-_\.]', '_', name)  # Keep only word chars, spaces, hyphens, underscores, dots
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'_{2,}', '_', name)  # Replace multiple underscores with single
    name = name.strip('_')  # Remove leading/trailing underscores
    return name[:100]  # Limit length to 100 chars


def generate_ticker_charts(raw_df, stats):
    """Generate one PNG chart per Ticker showing time series with roll points."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating Ticker charts")
    print("=" * 60)
    
    if raw_df.empty:
        print("  No data for charts")
        return
    
    unique_tickers = raw_df['ticker_id'].unique()
    print(f"  Generating {len(unique_tickers)} charts...")
    
    generated = 0
    skipped = 0
    
    for i, ticker_id in enumerate(unique_tickers):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(unique_tickers)}")
        
        # Get data for this ticker
        ticker_data = raw_df[raw_df['ticker_id'] == ticker_id].sort_values('date')
        if ticker_data.empty:
            skipped += 1
            continue
            
        ticker_name = ticker_data['ticker_name'].iloc[0]
        ticker_stats = stats.get(ticker_id, {})
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot main time series
        ax.plot(ticker_data['date'], ticker_data['close'], 
                linewidth=1.5, color='#2c3e50', label='Price')
        
        # Mark roll points
        rolls = ticker_data[ticker_data['is_roll']]
        if len(rolls) > 0:
            ax.scatter(rolls['date'], rolls['close'], 
                      color='#e74c3c', s=50, zorder=5, label='Roll Points')
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(ticker_name, fontsize=14, fontweight='bold')
        
        # Subtitle with stats
        duration = ticker_stats.get('duration_days', 0)
        data_points = ticker_stats.get('data_points', 0)
        roll_count = ticker_stats.get('roll_count', 0)
        subtitle = f"Duration: {duration} days | Data points: {data_points:,} | Rolls: {roll_count}"
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
                ha='center', va='top', fontsize=10, style='italic')
        
        # Date formatting
        if len(ticker_data) > 365:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Legend if there are roll points
        if len(rolls) > 0:
            ax.legend()
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with sanitized filename
        filename = sanitize_filename(ticker_name) + '.png'
        filepath = CHARTS_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated += 1
    
    print(f"  Generated: {generated} charts")
    print(f"  Skipped: {skipped} tickers")


# =============================================================================
# STEP 5: Write Excel with one sheet per ticker
# =============================================================================

def write_ticker_excel(raw_df, stats, markets, category_counts):
    """Write results.xlsx with one sheet per Ticker plus Summary."""
    print("\n" + "=" * 60)
    print("STEP 5: Writing results.xlsx")
    print("=" * 60)
    
    xlsx_path = OUTPUT_DIR / 'results.xlsx'
    
    if raw_df.empty:
        print("  No time series data for Excel")
        return
    
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Summary sheet first
        summary_rows = []
        for ticker_id, s in stats.items():
            summary_rows.append({
                'Ticker ID': ticker_id,
                'Ticker Name': s['ticker_name'],
                'Duration (days)': s['duration_days'], 
                'Data Points': s['data_points'],
                'Rolls': s['roll_count'],
                'Start Date': s['start_date'],
                'End Date': s['end_date'],
            })
        
        summary_df = pd.DataFrame(summary_rows).sort_values('Data Points', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  ✓ Summary sheet with {len(summary_df)} tickers")
        
        # One sheet per ticker
        unique_tickers = raw_df['ticker_id'].unique()
        print(f"  Creating {len(unique_tickers)} ticker sheets...")
        
        created = 0
        skipped = 0
        
        for i, ticker_id in enumerate(unique_tickers):
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(unique_tickers)}")
            
            # Get data for this ticker
            ticker_data = raw_df[raw_df['ticker_id'] == ticker_id].sort_values('date')
            if ticker_data.empty:
                skipped += 1
                continue
            
            ticker_name = ticker_data['ticker_name'].iloc[0]
            
            # Prepare sheet data
            sheet_data = ticker_data[['date', 'close', 'volume', 'cusip', 'is_roll']].copy()
            sheet_data.columns = ['Date', 'Price', 'Volume', 'Active CUSIP', 'Is Roll Point']
            
            # Sheet name (Excel limits to 31 chars)
            sheet_name = sanitize_filename(ticker_name)[:31]
            
            # Handle duplicate sheet names
            base_sheet_name = sheet_name
            counter = 1
            while sheet_name in [ws.title for ws in writer.book.worksheets]:
                sheet_name = f"{base_sheet_name[:28]}_{counter}"
                counter += 1
            
            try:
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                created += 1
            except Exception as e:
                print(f"  Warning: Could not create sheet for {ticker_name}: {e}")
                skipped += 1
        
        print(f"  ✓ Created {created} ticker sheets")
        if skipped > 0:
            print(f"  ✗ Skipped {skipped} sheets")
    
    print(f"  ✓ {xlsx_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PREDICTION MARKET TICKER TIME SERIES")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Step 1: Load markets
    markets, filtered, category_counts = load_markets()
    
    # Step 2: Build ticker mapping
    mapping_df, ticker_chains = build_ticker_mapping(markets)
    
    # Step 3: Build time series
    raw_df, adj_df, stats = build_ticker_timeseries(ticker_chains, markets)
    
    # Step 4: Generate individual ticker charts
    generate_ticker_charts(raw_df, stats)
    
    # Step 5: Write Excel with one sheet per ticker
    write_ticker_excel(raw_df, stats, markets, category_counts)
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    if stats:
        print(f"  Total Tickers with time series: {len(stats)}")
        print(f"  Charts generated: {len([f for f in CHARTS_DIR.glob('*.png')])}")
        print(f"  Excel sheets: {len(stats) + 1} (1 summary + {len(stats)} tickers)")
        print(f"  Total data points: {sum(s['data_points'] for s in stats.values()):,}")
        print(f"  Total roll points: {sum(s['roll_count'] for s in stats.values())}")
        
        # Date range
        start_dates = [s['start_date'] for s in stats.values()]
        end_dates = [s['end_date'] for s in stats.values()]
        print(f"  Data range: {min(start_dates)} to {max(end_dates)}")
    else:
        print("  No time series data generated")
    
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()