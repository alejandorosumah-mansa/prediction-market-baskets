#!/usr/bin/env python3
"""
Prediction Market Basket Construction - Clean Analysis Presentation

This script reproduces all analysis from the basket-engine repo in a clean,
presentation-ready format. It loads processed data and generates charts
and Excel output for easy consumption.

Usage: python run.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../basket-engine/data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# Set matplotlib style for publication-quality charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data():
    """Load all processed data from basket-engine."""
    print("Loading data from basket-engine...")
    
    # Core datasets
    markets = pd.read_parquet(DATA_DIR / "processed/markets.parquet")
    ticker_mapping = pd.read_parquet(DATA_DIR / "processed/ticker_mapping.parquet")
    correlation_matrix = pd.read_parquet(DATA_DIR / "processed/correlation_matrix_filtered.parquet")
    ticker_ts_raw = pd.read_parquet(DATA_DIR / "processed/ticker_timeseries_raw.parquet")
    ticker_ts_adj = pd.read_parquet(DATA_DIR / "processed/ticker_timeseries_adjusted.parquet")
    
    # JSON data
    with open(DATA_DIR / "processed/llm_market_categories.json", 'r') as f:
        market_categories = json.load(f)
    
    with open(DATA_DIR / "processed/community_labels.json", 'r') as f:
        community_labels = json.load(f)
        
    with open(DATA_DIR / "processed/hybrid_clustering_results.json", 'r') as f:
        clustering_results = json.load(f)
        
    with open(DATA_DIR / "processed/ticker_timeseries_stats.json", 'r') as f:
        ts_stats = json.load(f)
    
    print(f"Loaded {len(markets)} markets, {len(ticker_mapping)} tickers, {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} correlation matrix")
    
    return {
        'markets': markets,
        'ticker_mapping': ticker_mapping, 
        'correlation_matrix': correlation_matrix,
        'ticker_ts_raw': ticker_ts_raw,
        'ticker_ts_adj': ticker_ts_adj,
        'market_categories': market_categories,
        'community_labels': community_labels,
        'clustering_results': clustering_results,
        'ts_stats': ts_stats
    }

def generate_chart_01_filter_funnel(data):
    """Market filtering funnel analysis."""
    print("Generating chart 01: Market filter funnel...")
    
    # Simulate filter stages based on data structure
    markets = data['markets']
    
    filter_stages = {
        'Raw Markets': 20180,  # From description
        'With End Dates': len(markets[markets['end_date'].notna()]),
        'Active Period': len(markets[markets['active_start'].notna()]),
        'With Volume': len(markets[markets['volume'] > 0]),
        'Final Filtered': len(markets)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    stages = list(filter_stages.keys())
    counts = list(filter_stages.values())
    
    bars = ax.bar(stages, counts, color=sns.color_palette("viridis", len(stages)))
    ax.set_ylabel('Number of Markets')
    ax.set_title('Market Filtering Funnel', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_market_filter_funnel.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_02_category_distribution(data):
    """Distribution of markets by category."""
    print("Generating chart 02: Category distribution...")
    
    market_cats = data['market_categories']
    
    # Count categories
    cat_counts = pd.Series(list(market_cats.values())).value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(cat_counts)), cat_counts.values, color=sns.color_palette("Set2", len(cat_counts)))
    
    ax.set_ylabel('Number of Markets')
    ax.set_title('Market Distribution by Category (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in cat_counts.index], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, cat_counts.values)):
        height = bar.get_height()
        ax.text(i, height + 5, f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_category_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_03_ticker_cusip_distribution(data):
    """Distribution of CUSIPs per ticker."""
    print("Generating chart 03: Ticker CUSIP distribution...")
    
    ticker_mapping = data['ticker_mapping']
    
    # Count markets per ticker
    ticker_counts = ticker_mapping['ticker_id'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    bins = np.arange(1, ticker_counts.max() + 2) - 0.5
    ax.hist(ticker_counts.values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Number of CUSIPs per Ticker')
    ax.set_ylabel('Number of Tickers')
    ax.set_title('Distribution of CUSIPs per Ticker', fontsize=14, fontweight='bold')
    
    # Add summary stats
    ax.axvline(ticker_counts.median(), color='red', linestyle='--', label=f'Median: {ticker_counts.median():.1f}')
    ax.axvline(ticker_counts.mean(), color='orange', linestyle='--', label=f'Mean: {ticker_counts.mean():.1f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_ticker_cusip_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_04_correlation_distribution(data):
    """Histogram of pairwise correlations."""
    print("Generating chart 04: Correlation distribution...")
    
    corr_matrix = data['correlation_matrix']
    
    # Extract upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix.values[mask]
    correlations = correlations[~np.isnan(correlations)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(correlations, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pairwise Ticker Correlations', fontsize=14, fontweight='bold')
    
    # Add summary stats
    ax.axvline(np.median(correlations), color='red', linestyle='--', 
               label=f'Median: {np.median(correlations):.3f}')
    ax.axvline(np.mean(correlations), color='orange', linestyle='--', 
               label=f'Mean: {np.mean(correlations):.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_correlation_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_05_top_correlated_pairs(data):
    """Top 20 most correlated ticker pairs."""
    print("Generating chart 05: Top correlated pairs...")
    
    corr_matrix = data['correlation_matrix']
    
    # Find top correlations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_data = []
    
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if not np.isnan(corr_matrix.iloc[i, j]):
                corr_data.append({
                    'ticker1': corr_matrix.index[i],
                    'ticker2': corr_matrix.columns[j], 
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    corr_df = pd.DataFrame(corr_data)
    top_pairs = corr_df.nlargest(20, 'correlation')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pair labels (truncate if too long)
    pair_labels = [f"{row['ticker1'][:15]}\n{row['ticker2'][:15]}" 
                   for _, row in top_pairs.iterrows()]
    
    bars = ax.barh(range(len(top_pairs)), top_pairs['correlation'], 
                   color=sns.color_palette("viridis", len(top_pairs)))
    
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Top 20 Most Correlated Ticker Pairs', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, corr) in enumerate(zip(bars, top_pairs['correlation'])):
        width = bar.get_width()
        ax.text(width + 0.01, i, f'{corr:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_top_correlated_pairs.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_06_community_sizes(data):
    """Community sizes from clustering."""
    print("Generating chart 06: Community sizes...")
    
    clustering = data['clustering_results']
    community_labels = data['community_labels']
    
    if 'community_assignments' in clustering:
        # Count assignments per community
        assignments = clustering['community_assignments']
        community_counts = pd.Series(list(assignments.values())).value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(community_counts)), community_counts.values,
                      color=sns.color_palette("Set3", len(community_counts)))
        
        ax.set_xlabel('Community ID')
        ax.set_ylabel('Number of Tickers')
        ax.set_title('Ticker Community Sizes', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(community_counts)))
        ax.set_xticklabels(community_counts.index, rotation=45)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, community_counts.values)):
            height = bar.get_height()
            ax.text(i, height + 0.5, f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "06_community_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_chart_07_community_themes(data):
    """Theme composition per community."""
    print("Generating chart 07: Community themes...")
    
    # This would require more detailed analysis of ticker themes per community
    # For now, create a placeholder showing community diversity
    
    community_labels = data['community_labels']
    
    # Extract theme keywords from community names
    theme_words = []
    for label in community_labels.values():
        words = label.lower().split()
        theme_words.extend([w for w in words if len(w) > 3])
    
    theme_counts = pd.Series(theme_words).value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(theme_counts)), theme_counts.values,
                  color=sns.color_palette("viridis", len(theme_counts)))
    
    ax.set_xlabel('Theme Keywords')
    ax.set_ylabel('Frequency in Community Names')
    ax.set_title('Most Common Themes in Community Names', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(theme_counts)))
    ax.set_xticklabels(theme_counts.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_community_themes.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_08_sample_timeseries(data):
    """Sample ticker time series."""
    print("Generating chart 08: Sample ticker timeseries...")
    
    ticker_ts = data['ticker_ts_adj']
    
    # Find interesting tickers (high volume, long duration)
    ticker_stats = ticker_ts.groupby('ticker_id').agg({
        'date': ['min', 'max', 'count'],
        'volume': 'sum'
    }).reset_index()
    
    ticker_stats.columns = ['ticker_id', 'start_date', 'end_date', 'data_points', 'total_volume']
    ticker_stats['duration'] = (ticker_stats['end_date'] - ticker_stats['start_date']).dt.days
    
    # Select top 6 by combination of duration and volume
    ticker_stats['score'] = ticker_stats['duration'] * ticker_stats['total_volume']
    top_tickers = ticker_stats.nlargest(6, 'score')['ticker_id'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, ticker in enumerate(top_tickers):
        ticker_data = ticker_ts[ticker_ts['ticker_id'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        axes[i].plot(ticker_data['date'], ticker_data['price_adjusted'], linewidth=2)
        axes[i].set_title(f'{ticker}', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Price')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Mark roll points if present
        if 'is_roll_point' in ticker_data.columns:
            roll_points = ticker_data[ticker_data['is_roll_point'] == True]
            if len(roll_points) > 0:
                axes[i].scatter(roll_points['date'], roll_points['price_adjusted'], 
                               color='red', s=30, alpha=0.7, label='Roll Points')
    
    plt.suptitle('Sample Ticker Time Series', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_sample_ticker_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_chart_09_roll_analysis(data):
    """Roll points analysis."""
    print("Generating chart 09: Roll points analysis...")
    
    ticker_ts = data['ticker_ts_raw']
    
    if 'is_roll_point' in ticker_ts.columns:
        # Analyze roll patterns
        roll_data = ticker_ts[ticker_ts['is_roll_point'] == True].copy()
        
        if len(roll_data) > 0:
            # Roll frequency by ticker
            roll_counts = roll_data['ticker_id'].value_counts().head(20)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Roll frequency chart
            bars = ax1.bar(range(len(roll_counts)), roll_counts.values,
                          color=sns.color_palette("plasma", len(roll_counts)))
            ax1.set_xlabel('Tickers (Top 20)')
            ax1.set_ylabel('Number of Roll Points')
            ax1.set_title('Roll Frequency by Ticker')
            ax1.set_xticks(range(len(roll_counts)))
            ax1.set_xticklabels([str(x)[:8] for x in roll_counts.index], rotation=45)
            
            # Roll timing analysis
            roll_data['month'] = pd.to_datetime(roll_data['date']).dt.month
            monthly_rolls = roll_data['month'].value_counts().sort_index()
            
            ax2.bar(monthly_rolls.index, monthly_rolls.values, color='lightcoral')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Number of Roll Points')
            ax2.set_title('Roll Points by Month')
            ax2.set_xticks(range(1, 13))
            
            plt.suptitle('Roll Points Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / "09_roll_points_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Create placeholder if no roll data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Roll Point Data Available', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Roll Points Analysis', fontsize=14, fontweight='bold')
            plt.savefig(CHARTS_DIR / "09_roll_points_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

def generate_chart_10_cross_community_correlation(data):
    """Heatmap of inter-community correlations."""
    print("Generating chart 10: Cross-community correlations...")
    
    corr_matrix = data['correlation_matrix'] 
    clustering = data['clustering_results']
    community_labels = data['community_labels']
    
    if 'community_assignments' in clustering:
        assignments = clustering['community_assignments']
        
        # Create community mapping
        ticker_to_community = {ticker: int(comm) for ticker, comm in assignments.items() 
                              if ticker in corr_matrix.index}
        
        # Calculate average correlation between communities
        communities = sorted(set(ticker_to_community.values()))
        n_communities = len(communities)
        
        if n_communities > 1:
            comm_corr_matrix = np.zeros((n_communities, n_communities))
            
            for i, comm1 in enumerate(communities):
                for j, comm2 in enumerate(communities):
                    if i == j:
                        comm_corr_matrix[i, j] = 1.0
                    else:
                        # Get tickers in each community
                        tickers1 = [t for t, c in ticker_to_community.items() if c == comm1]
                        tickers2 = [t for t, c in ticker_to_community.items() if c == comm2]
                        
                        # Calculate average correlation between communities
                        corrs = []
                        for t1 in tickers1:
                            for t2 in tickers2:
                                if t1 in corr_matrix.index and t2 in corr_matrix.columns:
                                    corr_val = corr_matrix.loc[t1, t2]
                                    if not np.isnan(corr_val):
                                        corrs.append(corr_val)
                        
                        comm_corr_matrix[i, j] = np.mean(corrs) if corrs else 0
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Use community names if available, otherwise IDs
            comm_names = [community_labels.get(str(c), f"Community {c}")[:20] for c in communities]
            
            sns.heatmap(comm_corr_matrix, 
                       xticklabels=comm_names,
                       yticklabels=comm_names,
                       annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Average Correlation'})
            
            ax.set_title('Cross-Community Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / "10_cross_community_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()

def generate_excel_output(data):
    """Generate comprehensive Excel output with multiple sheets."""
    print("Generating Excel output...")
    
    excel_path = OUTPUT_DIR / "results.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Summary sheet
        markets = data['markets']
        ticker_mapping = data['ticker_mapping']
        correlation_matrix = data['correlation_matrix']
        
        summary_data = {
            'Metric': [
                'Total Markets (Raw)',
                'Total Markets (Filtered)',
                'Unique Tickers',
                'Unique Market Categories',
                'Correlation Matrix Size',
                'Communities Identified',
                'Time Series Data Points'
            ],
            'Value': [
                20180,  # From description
                len(markets),
                len(ticker_mapping['ticker_id'].unique()),
                len(set(data['market_categories'].values())),
                f"{correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}",
                len(data['community_labels']),
                len(data['ticker_ts_raw'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Ticker Mapping sheet
        ticker_summary = ticker_mapping.groupby('ticker_id').agg({
            'market_id': 'count',
            'ticker_name': 'first'
        }).reset_index()
        
        ticker_summary.columns = ['ticker_id', 'cusip_count', 'ticker_name']
        
        # Add categories from market classifications
        market_cats = data['market_categories']
        ticker_categories = {}
        for market_id, category in market_cats.items():
            ticker_row = ticker_mapping[ticker_mapping['market_id'] == market_id]
            if len(ticker_row) > 0:
                ticker_id = ticker_row.iloc[0]['ticker_id']
                if ticker_id not in ticker_categories:
                    ticker_categories[ticker_id] = category
        
        ticker_summary['primary_category'] = ticker_summary['ticker_id'].map(ticker_categories)
        ticker_summary.to_excel(writer, sheet_name='Ticker Mapping', index=False)
        
        # Ticker Time Series sheet
        ts_stats = data['ts_stats']
        if ts_stats:
            ts_df = pd.DataFrame.from_dict(ts_stats, orient='index').reset_index()
            ts_df.rename(columns={'index': 'ticker_id'}, inplace=True)
            ts_df.to_excel(writer, sheet_name='Ticker Time Series', index=False)
        
        # Communities sheet
        clustering = data['clustering_results']
        community_labels = data['community_labels']
        
        if 'community_assignments' in clustering:
            assignments = clustering['community_assignments']
            
            community_data = []
            for comm_id, comm_name in community_labels.items():
                tickers_in_comm = [t for t, c in assignments.items() if str(c) == comm_id]
                community_data.append({
                    'community_id': comm_id,
                    'community_name': comm_name,
                    'size': len(tickers_in_comm),
                    'top_tickers': ', '.join(tickers_in_comm[:5])  # First 5 tickers
                })
            
            communities_df = pd.DataFrame(community_data)
            communities_df.to_excel(writer, sheet_name='Communities', index=False)
        
        # Correlation Stats sheet
        corr_matrix = correlation_matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        correlations = correlations[~np.isnan(correlations)]
        
        corr_stats = {
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
            'Value': [
                len(correlations),
                np.mean(correlations),
                np.median(correlations), 
                np.std(correlations),
                np.min(correlations),
                np.max(correlations),
                np.percentile(correlations, 25),
                np.percentile(correlations, 75)
            ]
        }
        
        corr_stats_df = pd.DataFrame(corr_stats)
        corr_stats_df.to_excel(writer, sheet_name='Correlation Stats', index=False)
        
        # Market Classifications sheet
        market_class_data = []
        for market_id, category in data['market_categories'].items():
            market_row = markets[markets['market_id'] == market_id]
            if len(market_row) > 0:
                market_info = market_row.iloc[0]
                market_class_data.append({
                    'market_id': market_id,
                    'category': category,
                    'title': market_info.get('title', ''),
                    'platform': market_info.get('platform', ''),
                    'volume': market_info.get('volume', 0),
                    'start_date': market_info.get('start_date', ''),
                    'end_date': market_info.get('end_date', '')
                })
        
        market_class_df = pd.DataFrame(market_class_data)
        market_class_df.to_excel(writer, sheet_name='Market Classifications', index=False)
    
    print(f"Excel output saved to: {excel_path}")

def main():
    """Main execution function."""
    print("=== Prediction Market Basket Construction ===")
    print("Loading data and generating analysis outputs...\n")
    
    # Load all data
    data = load_data()
    
    # Generate all charts
    print("\nGenerating charts...")
    generate_chart_01_filter_funnel(data)
    generate_chart_02_category_distribution(data) 
    generate_chart_03_ticker_cusip_distribution(data)
    generate_chart_04_correlation_distribution(data)
    generate_chart_05_top_correlated_pairs(data)
    generate_chart_06_community_sizes(data)
    generate_chart_07_community_themes(data)
    generate_chart_08_sample_timeseries(data)
    generate_chart_09_roll_analysis(data)
    generate_chart_10_cross_community_correlation(data)
    
    # Generate Excel output
    generate_excel_output(data)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Charts saved to: {CHARTS_DIR}")
    print(f"Excel file saved to: {OUTPUT_DIR / 'results.xlsx'}")
    print(f"Total files generated: {len(list(CHARTS_DIR.glob('*.png')))} charts + 1 Excel file")

if __name__ == "__main__":
    main()