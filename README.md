# Prediction Market Ticker Generation

Build continuous time series from prediction market contracts by mapping individual contracts (CUSIPs) to recurring concepts (Tickers) and chaining them across expirations.

## The Problem

Prediction market contracts are ephemeral. "Will the Fed cut 25bps in March 2026?" expires in March. "Will the Fed cut 25bps in June 2026?" is a different contract. But they ask the same recurring question. To do any meaningful time series analysis, you need to chain these together, like futures continuous contracts.

## The Taxonomy

```
Theme ("Central Banks & Monetary Policy")
  └── Event ("Fed Rate Decision")
       └── Ticker ("Will Fed cut 25bps?")          ← recurring concept, no expiration
            └── CUSIP ("Will Fed cut 25bps in March 2026?")  ← specific contract with expiration
```

- **CUSIP**: Individual contract with an expiration date. What Polymarket calls a "market."
- **Ticker**: The recurring question that spawns new CUSIPs over time.
- **Event**: Groups related Tickers (e.g., different rate outcomes for the same meeting).
- **Theme**: Macro category (elections, crypto, geopolitics, etc.).

## What This Does

`python run.py` takes ~20K raw prediction market contracts and:

1. **Classifies** markets into 19 themes (LLM, cached)
2. **Filters** out sports and entertainment (60%+ of all markets)
3. **Maps CUSIPs → Tickers** using regex normalization + exact matching
4. **Builds continuous time series** for each rollable Ticker:
   - **Raw levels**: front-month rolling, price jumps at roll points preserved
   - **Adjusted series**: return-chained (like adjusted close), continuous for correlation work
5. **Outputs** everything to `output/`

## Output

```
output/
├── charts/
│   ├── 01_category_distribution.png      # Markets by theme
│   ├── 02_ticker_cusip_distribution.png  # CUSIPs per Ticker
│   ├── 03_sample_timeseries_fed.png      # Fed rate Ticker series
│   ├── 04_sample_timeseries_crypto.png   # Bitcoin/crypto Ticker series
│   ├── 05_sample_timeseries_politics.png # Election Ticker series
│   ├── 06_sample_timeseries_geopolitics.png # Iran/conflict series
│   ├── 07_roll_points_analysis.png       # Roll behavior across Tickers
│   └── 08_coverage_timeline.png          # Data coverage over time
└── results.xlsx
    ├── Summary                           # Pipeline stats
    ├── Ticker Mapping                    # All Tickers with CUSIP counts
    ├── Ticker Chains                     # Roll chains per Ticker
    ├── Time Series Stats                 # Duration, rolls, data points per Ticker
    └── Market Classifications            # All 20K markets categorized
```

## How to Run

```bash
python run.py
```

Requires: `pandas`, `numpy`, `matplotlib`, `openpyxl`, `openai` (for classification cache)

Data source: `../basket-engine/data/` (raw market data and processed caches)

## Key Results

- **20,180 markets** ingested from Polymarket + Kalshi
- **16,840 unique Tickers** identified
- **2,009 rollable Tickers** (2+ CUSIPs)
- **255 Tickers** with continuous time series (30+ days)
- **41,539 data points** spanning Jan 2024 – Feb 2026
- **228 roll points** identified and handled

## Methodology

### Ticker Mapping
1. Strip time-specific parts from titles (months, years, meeting dates) via regex
2. Normalize format variants ("Will the Fed decrease" → "Fed decrease", "$150K" → "$150,000")
3. Exact string match on normalized titles (no fuzzy matching to avoid merging different outcomes)
4. Sort CUSIPs by expiration date to build roll chains

### Continuous Time Series
- **Front-month rolling**: always hold the nearest-expiry active CUSIP
- **Roll trigger**: contract resolves or reaches expiration
- **Adjusted series**: daily probability changes chained continuously (returns, not levels)
- **Roll day treatment**: return = 0 at roll point (avoids artificial jumps)

## Limitations

- 71% of rollable Tickers lack candle data files (Polymarket data gap)
- Sports markets dominate by count (12K of 20K) but are filtered out
- Title format changes across years require ongoing regex maintenance
- "reach" vs "hit" in Bitcoin titles creates duplicate Tickers (edge case)
