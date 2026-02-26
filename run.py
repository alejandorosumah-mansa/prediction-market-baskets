#!/usr/bin/env python3
"""
Prediction Market Ticker Index Pipeline

Uses pre-processed parquet data to build index-based charts (base 100)
for the top 20 most modelable geopolitical/financial tickers.

Key features:
- Index construction (base 100) instead of raw probability
- Gap detection: breaks lines at >7 day gaps
- Better dedup: separates contracts by expiration year/cycle
- Geopolitical/war focus with minimum quota
- Modelability scoring favoring continuous data
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("../basket-engine/data/processed")
OUTPUT_DIR = Path("output")
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
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

# Categories
GEO_CONFLICT_CATEGORIES = {
    'russia_ukraine', 'middle_east', 'global_politics', 'china_geopolitics',
    'us_military', 'venezuela',
}

KEEP_CATEGORIES = GEO_CONFLICT_CATEGORIES | {
    'fed_monetary_policy', 'crypto_digital', 'us_elections', 'us_economic',
    'legal_regulatory', 'energy_commodities',
}

EXCLUDE_CATEGORIES = {
    'sports', 'entertainment',
}

GAP_THRESHOLD_DAYS = 7
CHAIN_GAP_THRESHOLD_DAYS = 30
MIN_DATA_POINTS = 30
MIN_GEO_IN_TOP20 = 10


def load_data():
    """Load all parquet files."""
    print("Loading data...")
    ts = pd.read_parquet(DATA_DIR / "ticker_timeseries_raw.parquet")
    tm = pd.read_parquet(DATA_DIR / "ticker_mapping.parquet")
    mc = pd.read_parquet(DATA_DIR / "market_classifications.parquet")
    markets = pd.read_parquet(DATA_DIR / "markets.parquet")
    
    ts['date'] = pd.to_datetime(ts['date'])
    tm['end_date_parsed'] = pd.to_datetime(tm['end_date_parsed'], errors='coerce')
    
    print(f"  Timeseries: {len(ts):,} rows, {ts['ticker_id'].nunique()} tickers")
    print(f"  Markets: {len(markets):,}")
    return ts, tm, mc, markets


def get_ticker_categories(tm, mc):
    """Get category for each ticker_id."""
    merged = tm.merge(mc[['market_id', 'category']], on='market_id', how='left')
    return merged.groupby('ticker_id')['category'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
    )


def check_chain_continuity(ts_data, max_gap_days=CHAIN_GAP_THRESHOLD_DAYS):
    """Check if a ticker's constituent contracts form a continuous chain.
    Returns True if no gap between consecutive contracts exceeds max_gap_days."""
    if len(ts_data) < 2:
        return True
    
    # Check gaps between different cusips
    sorted_data = ts_data.sort_values('date')
    cusip_changes = sorted_data['active_cusip'] != sorted_data['active_cusip'].shift(1)
    change_indices = sorted_data.index[cusip_changes & (sorted_data.index != sorted_data.index[0])]
    
    for idx in change_indices:
        prev_idx = sorted_data.index[sorted_data.index.get_loc(idx) - 1]
        gap = (sorted_data.loc[idx, 'date'] - sorted_data.loc[prev_idx, 'date']).days
        if gap > max_gap_days:
            return False
    return True


def compute_modelability(ts_data):
    """Compute modelability score for a ticker's timeseries.
    Score = data_points * time_span_days / (1 + num_gaps) / (1 + max_jump_size * 10)
    """
    if len(ts_data) < MIN_DATA_POINTS:
        return 0.0
    
    sorted_data = ts_data.sort_values('date')
    dates = sorted_data['date']
    prices = sorted_data['price']
    
    span_days = (dates.max() - dates.min()).days
    if span_days < 1:
        return 0.0
    
    # Count gaps >7 days
    day_diffs = dates.diff().dt.days
    num_gaps = (day_diffs > GAP_THRESHOLD_DAYS).sum()
    
    # Max single-day price jump
    price_changes = prices.diff().abs()
    max_jump = price_changes.max() if len(price_changes) > 1 else 0.0
    if pd.isna(max_jump):
        max_jump = 0.0
    
    score = len(ts_data) * span_days / (1 + num_gaps) / (1 + max_jump * 10)
    return score


def insert_gap_nans(dates, values, gap_days=GAP_THRESHOLD_DAYS):
    """Insert NaN values at gaps to break the line in matplotlib."""
    if len(dates) < 2:
        return dates, values
    
    new_dates = []
    new_values = []
    
    for i in range(len(dates)):
        new_dates.append(dates.iloc[i])
        new_values.append(values.iloc[i])
        
        if i < len(dates) - 1:
            gap = (dates.iloc[i + 1] - dates.iloc[i]).days
            if gap > gap_days:
                # Insert a NaN point to break the line
                mid = dates.iloc[i] + pd.Timedelta(days=1)
                new_dates.append(mid)
                new_values.append(np.nan)
    
    return pd.Series(new_dates), pd.Series(new_values, dtype=float)


def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*\[\]]', '_', name)
    name = re.sub(r'[^\w\s\-_.]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    return name.strip('_')[:100]


def main():
    print("=" * 60)
    print("  PREDICTION MARKET INDEX PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    ts, tm, mc, markets = load_data()
    ticker_cats = get_ticker_categories(tm, mc)
    
    # Get ticker names
    ticker_names = tm.groupby('ticker_id')['ticker_name'].first()
    ticker_titles = tm.groupby('ticker_id')['title'].first()
    ticker_market_counts = tm.groupby('ticker_id')['market_id'].count()
    
    # Filter categories
    tickers_in_ts = ts['ticker_id'].unique()
    print(f"\nFiltering {len(tickers_in_ts)} tickers...")
    
    kept_tickers = []
    excluded = 0
    for tid in tickers_in_ts:
        cat = ticker_cats.get(tid, 'unknown')
        if cat in EXCLUDE_CATEGORIES:
            excluded += 1
            continue
        if cat not in KEEP_CATEGORIES and cat != 'unknown':
            excluded += 1
            continue
        kept_tickers.append(tid)
    
    print(f"  Kept: {len(kept_tickers)}, Excluded: {excluded}")
    
    # Compute scores
    print("\nComputing modelability scores...")
    scores = {}
    ticker_meta = {}
    
    for tid in kept_tickers:
        data = ts[ts['ticker_id'] == tid].sort_values('date')
        if len(data) < MIN_DATA_POINTS:
            continue
        
        # Check chain continuity
        if not check_chain_continuity(data):
            continue
        
        score = compute_modelability(data)
        if score <= 0:
            continue
        
        cat = ticker_cats.get(tid, 'unknown')
        name = ticker_names.get(tid, tid)
        
        dates = data['date']
        day_diffs = dates.diff().dt.days
        num_gaps = int((day_diffs > GAP_THRESHOLD_DAYS).sum())
        max_jump = float(data['price'].diff().abs().max())
        
        scores[tid] = score
        ticker_meta[tid] = {
            'ticker_name': name,
            'category': cat,
            'data_points': len(data),
            'span_days': (dates.max() - dates.min()).days,
            'num_gaps': num_gaps,
            'max_jump': round(max_jump, 4),
            'num_contracts': int(data['active_cusip'].nunique()),
            'start_date': str(dates.min().date()),
            'end_date': str(dates.max().date()),
            'is_geo': cat in GEO_CONFLICT_CATEGORIES,
        }
    
    print(f"  Scoreable tickers: {len(scores)}")
    
    # Select top 20 with geo quota
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    geo_tickers = [(tid, s) for tid, s in ranked if ticker_meta[tid]['is_geo']]
    non_geo_tickers = [(tid, s) for tid, s in ranked if not ticker_meta[tid]['is_geo']]
    
    # Take top geo first to meet quota, then fill with best overall
    top20 = []
    geo_count = 0
    
    # First pass: add tickers in score order, ensuring geo quota
    geo_added = []
    non_geo_added = []
    
    for tid, s in ranked:
        if ticker_meta[tid]['is_geo']:
            geo_added.append((tid, s))
        else:
            non_geo_added.append((tid, s))
    
    # Ensure at least MIN_GEO_IN_TOP20 geo tickers
    # Take top geo tickers first
    for tid, s in geo_added[:MIN_GEO_IN_TOP20]:
        top20.append(tid)
    
    # Fill remaining slots with best overall (excluding already added)
    remaining = 20 - len(top20)
    all_remaining = [(tid, s) for tid, s in ranked if tid not in set(top20)]
    for tid, s in all_remaining[:remaining]:
        top20.append(tid)
    
    # If we don't have enough geo, just take what we have
    if len(top20) < 20:
        for tid, s in ranked:
            if tid not in set(top20):
                top20.append(tid)
            if len(top20) >= 20:
                break
    
    top20 = top20[:20]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TOP 20 TICKERS")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Score':>10} {'Cat':>25} {'Pts':>5} {'Span':>5} {'Gaps':>4} {'Name'}")
    print(f"{'-'*3} {'-'*10} {'-'*25} {'-'*5} {'-'*5} {'-'*4} {'-'*50}")
    
    geo_in_top = 0
    for i, tid in enumerate(top20, 1):
        m = ticker_meta[tid]
        s = scores[tid]
        geo_flag = '🌍' if m['is_geo'] else '  '
        if m['is_geo']:
            geo_in_top += 1
        print(f"{i:>3} {s:>10.0f} {m['category']:>25} {m['data_points']:>5} {m['span_days']:>5} {m['num_gaps']:>4} {geo_flag} {m['ticker_name'][:55]}")
    
    print(f"\nGeopolitical/conflict in top 20: {geo_in_top}")
    
    # Generate charts
    print(f"\nGenerating charts...")
    for f in CHARTS_DIR.glob('*.png'):
        f.unlink()
    
    for rank, tid in enumerate(top20, 1):
        data = ts[ts['ticker_id'] == tid].sort_values('date').reset_index(drop=True)
        m = ticker_meta[tid]
        
        # Build index (base 100)
        first_price = data['price'].iloc[0]
        if first_price <= 0:
            first_price = data['price'][data['price'] > 0].iloc[0] if (data['price'] > 0).any() else 0.01
        
        index_values = 100.0 * data['price'] / first_price
        
        # Insert NaN at gaps to break lines
        plot_dates, plot_values = insert_gap_nans(data['date'], index_values)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Main line
        ax.plot(plot_dates, plot_values, linewidth=1.2, color='#2c3e50', zorder=3)
        
        # Roll points as subtle vertical lines
        rolls = data[data['is_roll_point'] == True]
        for _, row in rolls.iterrows():
            ax.axvline(x=row['date'], color='#bdc3c7', linewidth=0.8,
                      linestyle=':', alpha=0.5, zorder=1)
        
        # Base 100 reference line
        ax.axhline(y=100, color='#95a5a6', linewidth=0.5, linestyle='-', alpha=0.3)
        
        # Y-axis
        ax.set_ylabel('Index Level (base = 100)', fontsize=13)
        
        # X-axis with prominent years
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', which='major', labelsize=14, length=8, width=1.5, pad=8)
        ax.tick_params(axis='x', which='minor', labelsize=9, length=4, colors='#888888')
        ax.set_xlabel('')
        
        # Title
        ax.set_title(m['ticker_name'], fontsize=15, fontweight='bold', pad=15)
        
        # Subtitle
        subtitle = (f"{m['category']} · {m['start_date']} → {m['end_date']} · "
                    f"{m['data_points']:,} pts · {m['num_contracts']} contracts · "
                    f"{m['num_gaps']} gaps")
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='#888888')
        
        # Grid
        ax.grid(True, which='major', alpha=0.15, linestyle='-')
        ax.grid(True, which='minor', alpha=0.08, linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = f"{rank:02d}_{sanitize_filename(m['ticker_name'])}.png"
        plt.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"  Generated {len(top20)} charts in {CHARTS_DIR}/")
    
    # Save results JSON
    results = {
        'top_20_tickers': [
            {'rank': i + 1, 'ticker_id': tid, 'score': round(scores[tid], 1), **ticker_meta[tid]}
            for i, tid in enumerate(top20)
        ],
        'total_scored_tickers': len(scores),
        'geo_in_top20': geo_in_top,
        'generated_at': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / 'top20_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to {OUTPUT_DIR}/top20_results.json")
    print("  DONE")


if __name__ == '__main__':
    main()
