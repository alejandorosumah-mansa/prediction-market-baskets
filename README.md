# Prediction Market Basket Construction

This repository presents a clean analysis of prediction market data to construct tradeable market baskets. The analysis processes over 20,000 prediction markets across multiple platforms to identify coherent themes and build correlation-based market clusters.

## Quick Start

```bash
python run.py
```

This single script reproduces the entire analysis and generates:
- **10 publication-quality charts** in `output/charts/`
- **Comprehensive Excel file** with 6 analysis sheets in `output/results.xlsx`

## What It Produces

### Charts Generated
1. **Market Filter Funnel** - How many markets survive each filtering stage
2. **Category Distribution** - Market counts by classified themes
3. **Ticker CUSIP Distribution** - Number of underlying contracts per ticker
4. **Correlation Distribution** - Histogram of pairwise ticker correlations
5. **Top Correlated Pairs** - Most correlated ticker pairs
6. **Community Sizes** - Number of tickers per identified cluster
7. **Community Themes** - Thematic analysis of clusters
8. **Sample Time Series** - Representative ticker price movements
9. **Roll Analysis** - Contract rollover patterns
10. **Cross-Community Correlations** - Inter-cluster correlation heatmap

### Excel Output Sheets
- **Summary** - Key metrics and data pipeline statistics
- **Ticker Mapping** - All tickers with CUSIP counts and categories
- **Ticker Time Series** - Statistical summary of each ticker's data
- **Communities** - Cluster definitions and top constituent tickers
- **Correlation Stats** - Distribution statistics of correlation matrix
- **Market Classifications** - All 20K markets with assigned categories

## Methodology

### Taxonomy Structure
The analysis uses a hierarchical classification system:
- **Theme** → Broad category (e.g., "Elections", "Crypto", "Fed Policy")
- **Event** → Specific occurrence (e.g., "2024 Presidential Election")
- **Ticker** → Tradeable instrument (e.g., "TRUMP2024")
- **CUSIP** → Individual market contract with expiry date

### Data Pipeline
Starting with **20,180 raw prediction markets**, the pipeline applies:

1. **Platform Integration** - Aggregates markets from Kalshi, Polymarket, Metaculus
2. **Date Filtering** - Removes markets without clear end dates
3. **Volume Filtering** - Excludes low-volume, illiquid markets
4. **Classification** - LLM-powered thematic categorization
5. **Ticker Mapping** - Groups related markets into common instruments
6. **Quality Control** - Validates data consistency and completeness

Final dataset contains **~18,000 high-quality markets** mapped to **~1,900 unique tickers**.

### Ticker Mapping Methodology
Markets are aggregated into tickers using:
- **Event commonality** - Markets about the same underlying event
- **Temporal coherence** - Contracts with related expiry dates
- **Semantic similarity** - NLP-based title and description matching
- **Platform reconciliation** - Cross-platform instrument identification

This creates continuous time series for each ticker by rolling between individual CUSIP contracts.

### Continuous Time Series Construction
For each ticker, the system:
1. **Identifies roll points** - When to switch between expiring contracts
2. **Price adjustment** - Maintains price continuity across rolls
3. **Volume aggregation** - Combines trading activity across contracts
4. **Gap filling** - Interpolates missing data points where appropriate

### Correlation Matrix (Quality Filters)
Pairwise correlations are computed with filters:
- **Minimum overlap** - Tickers must have ≥30 days of shared trading
- **Volume threshold** - Exclude periods with minimal trading activity
- **Outlier removal** - Filter extreme price movements that distort correlations
- **Stationarity checks** - Focus on price returns rather than levels

Final correlation matrix: **1,679 × 1,679 tickers** with high-quality overlap.

### Clustering Methodology
Market baskets are identified using:
- **Hybrid clustering** - Combines correlation-based and semantic approaches
- **Community detection** - Network-based algorithms on correlation graphs
- **Thematic coherence** - Validates clusters match intuitive market themes
- **Size balancing** - Ensures clusters are tradeable (not too large/small)

Result: **174 market communities** ranging from specific events (elections) to broad themes (crypto markets).

## Key Results

### Market Coverage
- **18,000+ markets** successfully processed and categorized
- **95%+ classification accuracy** via human validation sampling
- **1,900 unique tickers** identified across all platforms
- **174 coherent market communities** for basket construction

### Correlation Structure
- **Median correlation**: 0.089 (weak but meaningful relationships)
- **Strong correlations (>0.5)**: 8.3% of pairs (see Chart 5)
- **Negative correlations**: 23% of pairs (valuable for hedging)
- **Community correlation**: Within-cluster correlations 3x higher than between-cluster

### Time Series Quality
- **Average ticker duration**: 180 days of active trading
- **Roll frequency**: 2.3 rolls per ticker on average
- **Data completeness**: 87% daily coverage for active periods
- **Volume consistency**: Stable trading patterns across most tickers

### Notable Findings
- **Election markets** show highest internal correlation (0.34 median)
- **Crypto markets** exhibit strong momentum clustering
- **Fed policy markets** demonstrate clear policy cycles
- **Geopolitical events** cluster by geographic region
- **AI/Tech markets** emerging as distinct thematic category

*See Charts 6-10 for detailed community analysis and cross-correlations.*

## Limitations

### Data Constraints
- **Platform coverage** - Limited to major prediction market platforms
- **Historical depth** - Analysis covers 2023-2024 period only
- **Survivorship bias** - Focuses on markets with sufficient trading activity
- **Resolution accuracy** - Dependent on platform resolution mechanisms

### Methodological Limitations
- **Classification subjectivity** - LLM categorization may miss nuanced themes
- **Correlation assumptions** - Relationships may be non-linear or time-varying
- **Clustering stability** - Community assignments sensitive to parameter choices
- **Roll point identification** - Automated detection may miss optimal switch times

### Market Structure
- **Liquidity constraints** - Many tickers have limited trading depth
- **Platform fragmentation** - Same events may trade differently across platforms
- **Regulatory uncertainty** - Market availability subject to jurisdictional changes
- **Manipulation risk** - Small markets vulnerable to coordinated activity

## Future Enhancements

- **Real-time updating** - Live data feeds for dynamic basket rebalancing
- **Cross-platform arbitrage** - Exploit price differences between platforms  
- **Sentiment integration** - Incorporate news and social media signals
- **Risk model** - Value-at-Risk calculations for basket positions
- **Backtesting framework** - Historical performance analysis of basket strategies

---

*This analysis was generated from data processed in the companion `basket-engine` repository. The clean presentation here focuses on results and insights rather than implementation details.*