#!/usr/bin/env python3

import pandas as pd
from datetime import datetime

# Data collected from Reddit analysis
data = [
    {
        'Subreddit': 'r/Kalshi',
        'Link': 'https://www.reddit.com/r/Kalshi/comments/1rchjh3/made_rent_last_month_just_from_price_differences/',
        'User': 'Aggressive_Slide_169',
        'Question/Request': 'User discusses making money from price differences between Polymarket and Kalshi, mentions running "arbpoly" to catch arbitrage opportunities automatically. Wants portfolio/basket-like strategies that don\'t require predicting outcomes.',
        'Date': '2026-02-22',
        'Category': 'basket'
    },
    {
        'Subreddit': 'r/Polymarket', 
        'Link': 'https://www.reddit.com/r/Polymarket/comments/1ra84fo/how_i_research_kalshi_and_polymarket_markets/',
        'User': 'Such_Yogurtcloset392',
        'Question/Request': 'User discusses cross-platform analysis between Kalshi vs Polymarket for free edge, mentions building a system around market analysis. Asks about cross-platform arbitrage processes.',
        'Date': '2026-02-21',
        'Category': 'portfolio'
    },
    {
        'Subreddit': 'r/Polymarket',
        'Link': 'https://www.reddit.com/r/Polymarket/comments/1rbzw5c/been_running_an_ai_polymarket_experiment_for_a/',
        'User': 'Important_Opinion',
        'Question/Request': 'User running AI trading bot with infrastructure: scanner watching 2,000 markets, whale tracker on $1k+ moves, strategy engine. Looking for systematic approaches to prediction markets beyond individual trades.',
        'Date': '2026-02-22',
        'Category': 'portfolio'
    },
    {
        'Subreddit': 'r/Polymarket',
        'Link': 'https://www.reddit.com/r/Polymarket/comments/1rb6igw/opensource_tool_to_detect_the_incrementnonce/',
        'User': 'Vanadium_Hydroxide',
        'Question/Request': 'User built open-source tool "Nonce Guard" for detecting exploits and wants better infrastructure/tools for trading bots. Mentions need for portfolio protection tools.',
        'Date': '2026-02-22',
        'Category': 'portfolio'
    },
    {
        'Subreddit': 'r/Kalshi',
        'Link': 'https://www.reddit.com/r/Kalshi/comments/1rcvku6/combo_payout_flashed_9k_higher_than_it_should/',
        'User': 'GreatestThatNeverWas',
        'Question/Request': 'User was looking at Cavs win + over 213 combo bets, mentions wanting combined/basket-style betting strategies rather than individual market bets.',
        'Date': '2026-02-23',
        'Category': 'basket'
    },
    {
        'Subreddit': 'r/PredictionMarkets',
        'Link': 'https://www.reddit.com/r/PredictionMarkets/comments/1ralq01/dome_is_shutting_down_march_31_heres_what_to/',
        'User': 'pmxt_dev',
        'Question/Request': 'Developer discusses need for unified API across Polymarket, Kalshi, Limitless for cross-platform arb bots. Mentions pmxt as ccxt equivalent for prediction markets - infrastructure for basket/portfolio strategies.',
        'Date': '2026-02-21',
        'Category': 'ETF'
    },
    {
        'Subreddit': 'r/PredictionMarkets',
        'Link': 'https://www.reddit.com/r/PredictionMarkets/comments/1rcm7ga/finally_found_an_sdk_that_doesnt_feel_like_a/',
        'User': 'Nandhkumarr',
        'Question/Request': 'Developer working on arbitrage bot mentions advanced orders and multi-asset trading across timeframes. Looking for tools that enable portfolio-style trading rather than individual predictions.',
        'Date': '2026-02-23',
        'Category': 'portfolio'
    },
    {
        'Subreddit': 'r/PredictionMarkets',
        'Link': 'https://www.reddit.com/r/PredictionMarkets/comments/1rb65rp/merid/',
        'User': 'MaxExtractoor',
        'Question/Request': 'User built "MERID" - autonomous multi-AI swarm for trading Kalshi prediction markets. Uses grid of specialized AI agents across multiple assets and timeframes with unified order pipeline - essentially a prediction market ETF infrastructure.',
        'Date': '2026-02-22',
        'Category': 'ETF'
    },
    {
        'Subreddit': 'r/PredictionMarkets',
        'Link': 'https://www.reddit.com/r/PredictionMarkets/comments/1rbv2bn/buying_the_dips/',
        'User': 'Caseyyyy_',
        'Question/Request': 'User looking for strategy in Kalshi and Polymarket, mentions wanting to buy dips and wait for recovery - looking for portfolio-style approaches rather than single predictions.',
        'Date': '2026-02-22',
        'Category': 'portfolio'
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = '/Users/openclaw/.openclaw/workspace/prediction-market-baskets/output/reddit_research.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Excel file created: {output_file}")
print(f"Total entries: {len(data)}")
print("\nCategory breakdown:")
for category in df['Category'].unique():
    count = len(df[df['Category'] == category])
    print(f"  {category}: {count} entries")