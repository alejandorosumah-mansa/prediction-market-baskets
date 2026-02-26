#!/usr/bin/env python3
"""
Prediction Market Ticker Generation Pipeline

Takes ~20K raw prediction market contracts, maps them to recurring Tickers,
and builds continuous time series by rolling contracts across expirations.

Filters to geopolitical+financial contracts only, selects top 20 most modelable,
normalizes probabilities across contract rolls, and generates clean charts.

Usage: python run.py
Output: output/charts/*.png (top 20 modelable Tickers)
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

warnings.filterwarnings('ignore')

# Paths
BASKET_ENGINE = Path("../basket-engine/data")
OUTPUT_DIR = Path("output")
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot style - clean and professional
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
})

# Categories to KEEP (geopolitical + financial)
GEO_FINANCE_CATEGORIES = {
    'us_elections', 'crypto_digital', 'global_politics', 'us_economic',
    'middle_east', 'fed_monetary_policy', 'russia_ukraine', 'energy_commodities',
    'us_military', 'venezuela', 'china_geopolitics', 'legal_regulatory',
}

# =============================================================================
# STEP 1: Load and classify markets
# =============================================================================

def load_markets():
    """Load all markets, filter to geo+finance only."""
    print("=" * 60)
    print("STEP 1: Loading markets (geo+finance only)")
    print("=" * 60)

    markets = pd.read_parquet(BASKET_ENGINE / "processed/markets.parquet")
    print(f"  Total markets: {len(markets):,}")

    classifications = pd.read_parquet(BASKET_ENGINE / "processed/market_classifications.parquet")
    markets = markets.merge(classifications[['market_id', 'category']], on='market_id', how='left')

    # Filter to geo+finance only
    filtered = markets[markets['category'].isin(GEO_FINANCE_CATEGORIES)]
    print(f"  After geo+finance filter: {len(filtered):,}")

    category_counts = filtered['category'].value_counts()
    for cat, count in category_counts.items():
        print(f"    {cat}: {count:,}")

    return markets, filtered, category_counts


# =============================================================================
# STEP 2: Ticker mapping
# =============================================================================

def normalize_title(title):
    """Normalize market title to recurring Ticker concept."""
    if pd.isna(title):
        return ""

    n = str(title)

    n = re.sub(r'\bafter\s+the\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+meeting\b',
               'after meeting', n, flags=re.I)
    n = re.sub(r'\bafter\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+meeting\b',
               'after meeting', n, flags=re.I)
    n = re.sub(r'\bby\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
               'by [timeframe]', n, flags=re.I)
    n = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', '', n, flags=re.I)
    n = re.sub(r'\b20[2-9]\d\b', '', n)
    n = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', '', n, flags=re.I)
    n = re.sub(r'\bQ[1-4]\b', '', n, flags=re.I)
    n = re.sub(r'\bafter\s+(the\s+)?meeting\b', 'after meeting', n, flags=re.I)
    n = re.sub(r'\bbefore\s+20[2-9]\d\b', 'before [year]', n, flags=re.I)
    n = re.sub(r'\bby\s+20[2-9]\d\b', 'by [year]', n, flags=re.I)
    n = re.sub(r'\b20[2-9]\d[-\u2013]20[2-9]\d\b', '[daterange]', n)
    n = re.sub(r'\$(\d+)[kK]\b', lambda m: f'${m.group(1)},000', n)
    n = re.sub(r'^Will\s+the\s+', '', n, flags=re.I)
    n = re.sub(r'^Will\s+', '', n, flags=re.I)
    n = re.sub(r'\bdecreases\b', 'decrease', n, flags=re.I)
    n = re.sub(r'\bincreases\b', 'increase', n, flags=re.I)
    n = re.sub(r'\breaches\b', 'reach', n, flags=re.I)
    n = re.sub(r'\bwins\b', 'win', n, flags=re.I)
    n = re.sub(r'\s+in\s*\?\s*$', '?', n)
    n = re.sub(r'\s+in\s*$', '', n)
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
            'category': group['category'].mode().iloc[0] if not group['category'].mode().empty else '',
        }

    mapping_df = pd.DataFrame(ticker_mapping)

    total_tickers = len(ticker_chains)
    rollable = sum(1 for v in ticker_chains.values() if v['market_count'] >= 2)

    print(f"  Total Tickers: {total_tickers:,}")
    print(f"  Rollable (2+ CUSIPs): {rollable:,}")
    print(f"  Max CUSIPs per Ticker: {max(v['market_count'] for v in ticker_chains.values())}")

    return mapping_df, ticker_chains


# =============================================================================
# STEP 3: Build continuous time series with normalization
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
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                flat = []
                for sublist in data:
                    if isinstance(sublist, list):
                        flat.extend(sublist)
                data = flat
            if not data or not isinstance(data, list):
                return None
            rows = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                ts = item.get('end_period_ts')
                price = item.get('price', {})
                if isinstance(price, dict):
                    close_str = price.get('close_dollars', None)
                else:
                    close_str = None
                vol = item.get('volume', 0)
                if ts and close_str:
                    try:
                        rows.append({
                            'date': pd.to_datetime(int(ts), unit='s', errors='coerce'),
                            'close': float(close_str),
                            'volume': float(vol) if vol else 0,
                        })
                    except (ValueError, TypeError):
                        continue
            if not rows:
                try:
                    df = pd.DataFrame(data)
                    if 't' in df.columns:
                        df['date'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
                        df['close'] = pd.to_numeric(df.get('c', 0), errors='coerce')
                        df['volume'] = pd.to_numeric(df.get('v', 0), errors='coerce')
                    else:
                        return None
                except:
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


def normalize_probability_at_roll(combined, cusip_data):
    """
    Normalize probabilities across contract rolls using annualized rate.
    
    At each roll point, converts contract probabilities to annualized rates
    using: P_annualized = 1 - (1 - P_contract)^(365 / days_to_expiry)
    
    Then applies ratio adjustment so the series is continuous.
    """
    if len(combined) < 2:
        return combined
    
    combined = combined.copy()
    
    # Build a lookup: market_id -> end_date
    end_dates = {}
    for cd in cusip_data:
        end_dates[cd['market_id']] = cd['end_date']
    
    # Find roll points
    roll_indices = combined.index[combined['is_roll']].tolist()
    
    for ri in roll_indices:
        if ri == 0:
            continue
        
        prev_idx = ri - 1
        prev_close = combined.loc[prev_idx, 'close']
        curr_close = combined.loc[ri, 'close']
        
        prev_cusip = combined.loc[prev_idx, 'cusip']
        curr_cusip = combined.loc[ri, 'cusip']
        curr_date = combined.loc[ri, 'date']
        
        prev_end = end_dates.get(prev_cusip)
        curr_end = end_dates.get(curr_cusip)
        
        # Try annualized normalization
        if pd.notna(prev_end) and pd.notna(curr_end):
            prev_days = max((prev_end - curr_date).days, 1)
            curr_days = max((curr_end - curr_date).days, 1)
            
            # Convert both to annualized probability
            prev_ann = 1 - (1 - np.clip(prev_close, 0.001, 0.999)) ** (365.0 / prev_days)
            curr_ann = 1 - (1 - np.clip(curr_close, 0.001, 0.999)) ** (365.0 / curr_days)
            
            if curr_ann > 0.001:
                ratio = prev_ann / curr_ann
            else:
                ratio = 1.0
        else:
            # Fallback: simple ratio to eliminate jump
            if curr_close > 0.001:
                ratio = prev_close / curr_close
            else:
                ratio = 1.0
        
        # Clamp ratio to avoid extreme adjustments
        ratio = np.clip(ratio, 0.5, 2.0)
        
        # Apply ratio to all points from this roll onward until next roll
        # Find next roll point
        later_rolls = [r for r in roll_indices if r > ri]
        next_roll = later_rolls[0] if later_rolls else len(combined)
        
        mask = (combined.index >= ri) & (combined.index < next_roll)
        combined.loc[mask, 'close'] = combined.loc[mask, 'close'] * ratio
    
    # Clip to valid probability range
    combined['close'] = combined['close'].clip(0, 1)
    
    # Flag remaining jumps > 20% as data quality issues
    combined['daily_change'] = combined['close'].diff().abs()
    bad_jumps = combined['daily_change'] > 0.20
    num_bad = bad_jumps.sum()
    
    # Interpolate over bad jumps (not at roll points, those are already handled)
    bad_non_roll = bad_jumps & ~combined['is_roll']
    if bad_non_roll.any():
        combined.loc[bad_non_roll, 'close'] = np.nan
        combined['close'] = combined['close'].interpolate(method='linear')
        combined['close'] = combined['close'].ffill().bfill()
    
    combined = combined.drop(columns=['daily_change'], errors='ignore')
    return combined


def build_ticker_timeseries(ticker_chains, markets_df, min_days=30):
    """Build continuous time series for rollable Tickers with normalization."""
    print("\n" + "=" * 60)
    print("STEP 3: Building Ticker time series")
    print("=" * 60)

    rollable = {k: v for k, v in ticker_chains.items() if v['market_count'] >= 2}
    print(f"  Processing {len(rollable)} rollable Tickers...")

    all_raw = []
    stats = {}
    success = 0
    no_data = 0
    too_short = 0

    for i, (tid, chain) in enumerate(rollable.items()):
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(rollable)}")

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

        cusip_data.sort(key=lambda x: x['end_date'] if pd.notna(x['end_date']) else pd.Timestamp.max)

        combined_rows = []
        used_dates = set()

        for ci, cusip in enumerate(cusip_data):
            df = cusip['data']
            is_last = (ci == len(cusip_data) - 1)

            if is_last:
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
                end = cusip['end_date']
                for _, row in df.iterrows():
                    d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                    if d in used_dates:
                        continue
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

        combined['is_roll'] = combined['cusip'] != combined['cusip'].shift(1)
        combined.iloc[0, combined.columns.get_loc('is_roll')] = False

        # Store original roll points before normalization
        roll_count_raw = int(combined['is_roll'].sum())

        # Normalize probabilities across rolls
        combined = normalize_probability_at_roll(combined, cusip_data)

        combined['ticker_id'] = tid
        combined['ticker_name'] = chain['ticker_name']
        all_raw.append(combined)

        stats[tid] = {
            'ticker_name': chain['ticker_name'],
            'cusip_count': chain['market_count'],
            'data_points': len(combined),
            'duration_days': (combined['date'].max() - combined['date'].min()).days,
            'roll_count': roll_count_raw,
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
    print(f"    Total data points: {len(raw_df):,}")

    return raw_df, stats


# =============================================================================
# STEP 4: Select top 20 most modelable tickers
# =============================================================================

def select_top_modelable(raw_df, stats, top_n=20):
    """
    Rank tickers by modelability score and select top N.
    Score = data_points * (1 / (1 + num_jumps)) * time_span_days
    """
    print("\n" + "=" * 60)
    print(f"STEP 4: Selecting top {top_n} most modelable tickers")
    print("=" * 60)

    scores = {}
    for tid, s in stats.items():
        dp = s['data_points']
        rolls = s['roll_count']
        span = max(s['duration_days'], 1)

        # Compute smoothness: std of daily changes
        ticker_data = raw_df[raw_df['ticker_id'] == tid].sort_values('date')
        daily_changes = ticker_data['close'].diff().abs()
        smoothness = daily_changes.mean() if len(daily_changes) > 1 else 1.0
        smoothness = max(smoothness, 0.001)

        score = dp * (1.0 / (1 + rolls)) * span * (1.0 / (1 + smoothness * 10))
        scores[tid] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tids = [tid for tid, _ in ranked[:top_n]]

    print(f"\n  Top {top_n} tickers selected:")
    print(f"  {'Rank':<5} {'Score':>10} {'Days':>6} {'Pts':>6} {'Rolls':>6}  {'Ticker Name'}")
    print(f"  {'-'*5} {'-'*10} {'-'*6} {'-'*6} {'-'*6}  {'-'*40}")
    for rank, tid in enumerate(top_tids, 1):
        s = stats[tid]
        sc = scores[tid]
        print(f"  {rank:<5} {sc:>10.0f} {s['duration_days']:>6} {s['data_points']:>6} {s['roll_count']:>6}  {s['ticker_name'][:60]}")

    return top_tids


# =============================================================================
# STEP 5: Generate clean charts
# =============================================================================

def sanitize_filename(name):
    """Sanitize ticker name for use as filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[^\w\s\-_\.]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_')
    return name[:100]


def generate_ticker_charts(raw_df, stats, top_tids):
    """Generate one clean PNG chart per top ticker."""
    print("\n" + "=" * 60)
    print("STEP 5: Generating charts for top tickers")
    print("=" * 60)

    # Clear old charts
    for f in CHARTS_DIR.glob('*.png'):
        f.unlink()

    if raw_df.empty:
        print("  No data for charts")
        return

    print(f"  Generating {len(top_tids)} charts...")

    for rank, ticker_id in enumerate(top_tids, 1):
        ticker_data = raw_df[raw_df['ticker_id'] == ticker_id].sort_values('date')
        if ticker_data.empty:
            continue

        ticker_name = ticker_data['ticker_name'].iloc[0]
        ticker_stats = stats.get(ticker_id, {})

        fig, ax = plt.subplots(figsize=(14, 7))

        # Main line - raw data, no smoothing
        ax.plot(ticker_data['date'], ticker_data['close'],
                linewidth=1.2, color='#2c3e50', zorder=3)

        # Roll points - subtle
        rolls = ticker_data[ticker_data['is_roll']]
        if len(rolls) > 0:
            for _, roll_row in rolls.iterrows():
                ax.axvline(x=roll_row['date'], color='#bdc3c7', linewidth=0.8,
                          linestyle=':', alpha=0.6, zorder=1)

        # Y-axis
        ax.set_ylabel('Probability', fontsize=13)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # X-axis with prominent year labels
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

        # Make year labels prominent
        ax.tick_params(axis='x', which='major', labelsize=14, length=8, width=1.5, pad=8)
        ax.tick_params(axis='x', which='minor', labelsize=9, length=4, colors='#888888')

        ax.set_xlabel('', fontsize=1)  # No xlabel, year is self-explanatory

        # Title
        ax.set_title(ticker_name, fontsize=15, fontweight='bold', pad=15)

        # Subtitle with stats
        duration = ticker_stats.get('duration_days', 0)
        data_points = ticker_stats.get('data_points', 0)
        cat = ticker_stats.get('category', '')
        subtitle = f"{cat} · {duration}d span · {data_points:,} data points"
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='#888888')

        # Clean grid
        ax.grid(True, which='major', alpha=0.15, linestyle='-')
        ax.grid(True, which='minor', alpha=0.08, linestyle='-')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        filename = f"{rank:02d}_{sanitize_filename(ticker_name)}.png"
        filepath = CHARTS_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"  Generated: {len(top_tids)} charts in {CHARTS_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PREDICTION MARKET TICKER TIME SERIES")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Step 1: Load markets (geo+finance only)
    markets, filtered, category_counts = load_markets()

    # Step 2: Build ticker mapping (on filtered data)
    mapping_df, ticker_chains = build_ticker_mapping(filtered)

    # Step 3: Build time series with normalization
    raw_df, stats = build_ticker_timeseries(ticker_chains, markets)

    # Step 4: Select top 20 most modelable
    top_tids = select_top_modelable(raw_df, stats, top_n=20)

    # Step 5: Generate charts for top 20 only
    generate_ticker_charts(raw_df, stats, top_tids)

    # Save results
    results = {
        'top_20_tickers': [
            {'rank': i+1, 'ticker_id': tid, **stats[tid]}
            for i, tid in enumerate(top_tids)
        ],
        'total_geo_finance_tickers': len(stats),
        'generated_at': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / 'top20_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total geo+finance tickers with time series: {len(stats)}")
    print(f"  Top 20 charts generated in: {CHARTS_DIR}/")
    print(f"  Results saved to: {OUTPUT_DIR}/top20_results.json")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
