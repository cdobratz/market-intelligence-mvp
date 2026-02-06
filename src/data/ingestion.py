"""
Data Ingestion Module - API clients for financial data sources

This module provides functions to fetch data from:
- Alpha Vantage (stocks, forex, technical indicators)
- CoinGecko (cryptocurrency data)
- News API (financial news)
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class APIRateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass


class AlphaVantageClient:
    """
    Client for Alpha Vantage API
    Free tier: 25 API calls per day, 5 API calls per minute

    API Key: Get from https://www.alphavantage.co/support/#api-key
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided")

        self.rate_limit_delay = 12  # seconds between calls (5 per minute)

    def _make_request(self, params: dict[str, str]) -> dict[str, Any]:
        """Make API request with rate limiting"""
        params['apikey'] = self.api_key

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Check for rate limit or error messages
        if 'Error Message' in data:
            raise APIRateLimitError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise APIRateLimitError(f"Rate limit hit: {data['Note']}")

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        return data

    def get_daily_stock_data(
        self,
        symbol: str,
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Fetch daily stock data

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'GOOGL')
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching daily data for {symbol}...")

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        # Parse time series data
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            raise ValueError(f"No time series data found for {symbol}")

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Clean column names and convert to numeric
        df.columns = [col.split('. ')[1] for col in df.columns]
        df = df.astype(float)
        df['symbol'] = symbol

        return df

    def get_intraday_stock_data(
        self,
        symbol: str,
        interval: str = '5min'
    ) -> pd.DataFrame:
        """
        Fetch intraday stock data

        Args:
            symbol: Stock ticker
            interval: '1min', '5min', '15min', '30min', '60min'

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching intraday data for {symbol} at {interval} interval...")

        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'compact'
        }

        data = self._make_request(params)

        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise ValueError(f"No time series data found for {symbol}")

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = [col.split('. ')[1] for col in df.columns]
        df = df.astype(float)
        df['symbol'] = symbol

        return df

    def get_fx_daily(
        self,
        from_currency: str,
        to_currency: str,
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Fetch forex daily data

        Args:
            from_currency: e.g., 'EUR', 'GBP'
            to_currency: e.g., 'USD'
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with forex data
        """
        print(f"Fetching forex data for {from_currency}/{to_currency}...")

        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        time_series_key = 'Time Series FX (Daily)'
        if time_series_key not in data:
            raise ValueError(f"No forex data found for {from_currency}/{to_currency}")

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = [col.split('. ')[1] for col in df.columns]
        df = df.astype(float)
        df['pair'] = f"{from_currency}{to_currency}"

        return df

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = 'daily',
        time_period: int = 14,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch technical indicators

        Args:
            symbol: Stock ticker
            indicator: RSI, MACD, BBANDS, SMA, EMA, etc.
            interval: daily, weekly, monthly
            time_period: Number of data points
            **kwargs: Additional indicator-specific parameters

        Returns:
            DataFrame with indicator values
        """
        print(f"Fetching {indicator} for {symbol}...")

        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            **kwargs
        }

        data = self._make_request(params)

        # Find the technical analysis key
        tech_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                tech_key = key
                break

        if not tech_key:
            raise ValueError(f"No technical indicator data found for {symbol}")

        df = pd.DataFrame.from_dict(data[tech_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        df['symbol'] = symbol

        return df


class CoinGeckoClient:
    """
    Client for CoinGecko API
    Free tier: No API key required, rate limits apply

    Documentation: https://www.coingecko.com/en/api/documentation
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self.rate_limit_delay = 1.5  # seconds between calls

    def _make_request(self, endpoint: str, params: dict | None = None) -> Any:
        """Make API request with rate limiting"""
        url = f"{self.BASE_URL}/{endpoint}"

        response = requests.get(url, params=params or {})
        response.raise_for_status()

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        return response.json()

    def get_coin_list(self) -> list[dict[str, str]]:
        """Get list of all coins"""
        return self._make_request('coins/list')

    def get_coin_market_data(
        self,
        coin_ids: list[str],
        vs_currency: str = 'usd',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch market data for multiple coins

        Args:
            coin_ids: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
            vs_currency: Currency for price (usd, eur, etc.)
            days: Number of days of historical data

        Returns:
            DataFrame with market data
        """
        all_data = []

        for coin_id in coin_ids:
            print(f"Fetching market data for {coin_id}...")

            # Get current market data
            market_data = self._make_request(
                f"coins/{coin_id}/market_chart",
                params={'vs_currency': vs_currency, 'days': days}
            )

            # Convert to DataFrame
            prices = pd.DataFrame(
                market_data['prices'],
                columns=['timestamp', 'price']
            )
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices['coin_id'] = coin_id

            if 'market_caps' in market_data:
                market_caps = pd.DataFrame(
                    market_data['market_caps'],
                    columns=['timestamp', 'market_cap']
                )
                market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
                prices = prices.merge(market_caps, on='timestamp')

            if 'total_volumes' in market_data:
                volumes = pd.DataFrame(
                    market_data['total_volumes'],
                    columns=['timestamp', 'volume']
                )
                volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
                prices = prices.merge(volumes, on='timestamp')

            all_data.append(prices)

        return pd.concat(all_data, ignore_index=True)

    def get_trending_coins(self) -> list[dict]:
        """Get trending coins"""
        data = self._make_request('search/trending')
        return data.get('coins', [])


class NewsAPIClient:
    """
    Client for News API
    Free tier: 100 requests per day

    API Key: Get from https://newsapi.org/register
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("News API key not provided")

    def _make_request(self, endpoint: str, params: dict) -> dict[str, Any]:
        """Make API request"""
        url = f"{self.BASE_URL}/{endpoint}"
        params['apiKey'] = self.api_key

        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_everything(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        language: str = 'en',
        sort_by: str = 'relevancy',
        page_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news articles

        Args:
            query: Keywords or phrases to search for
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            sort_by: relevancy, popularity, publishedAt
            page_size: Number of results (max 100)

        Returns:
            DataFrame with articles
        """
        print(f"Fetching news for query: {query}...")

        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size
        }

        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request('everything', params)

        if data.get('status') != 'ok':
            raise ValueError(f"API error: {data.get('message')}")

        articles = data.get('articles', [])

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])

        return df

    def get_top_headlines(
        self,
        category: str = 'business',
        country: str = 'us',
        page_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch top headlines

        Args:
            category: business, entertainment, general, health, science, sports, technology
            country: 2-letter country code
            page_size: Number of results (max 100)

        Returns:
            DataFrame with headlines
        """
        print(f"Fetching top {category} headlines for {country}...")

        params = {
            'category': category,
            'country': country,
            'pageSize': page_size
        }

        data = self._make_request('top-headlines', params)

        if data.get('status') != 'ok':
            raise ValueError(f"API error: {data.get('message')}")

        articles = data.get('articles', [])

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])

        return df


# Utility functions for data storage
def save_to_parquet(
    df: pd.DataFrame,
    base_path: str,
    data_type: str,
    date: str
) -> str:
    """
    Save DataFrame to Parquet with partitioning

    Args:
        df: DataFrame to save
        base_path: Base directory path
        data_type: Type of data (stocks, forex, crypto, news)
        date: Date string (YYYY-MM-DD)

    Returns:
        Path to saved file
    """
    output_dir = Path(base_path) / data_type / f"date={date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{data_type}_{date}.parquet"
    df.to_parquet(output_file, index=True, compression='snappy')

    print(f"Data saved to: {output_file}")
    return str(output_file)


def save_to_json(
    data: Any,
    base_path: str,
    data_type: str,
    date: str
) -> str:
    """
    Save data to JSON

    Args:
        data: Data to save
        base_path: Base directory path
        data_type: Type of data
        date: Date string (YYYY-MM-DD)

    Returns:
        Path to saved file
    """
    output_dir = Path(base_path) / data_type / f"date={date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{data_type}_{date}.json"

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Data saved to: {output_file}")
    return str(output_file)


# Example usage
if __name__ == "__main__":
    # Test Alpha Vantage
    try:
        av_client = AlphaVantageClient()
        aapl_data = av_client.get_daily_stock_data('AAPL', outputsize='compact')
        print(f"Fetched {len(aapl_data)} days of AAPL data")
        print(aapl_data.head())
    except Exception as e:
        print(f"Alpha Vantage error: {e}")

    # Test CoinGecko
    try:
        cg_client = CoinGeckoClient()
        crypto_data = cg_client.get_coin_market_data(['bitcoin', 'ethereum'], days=7)
        print(f"Fetched {len(crypto_data)} crypto records")
        print(crypto_data.head())
    except Exception as e:
        print(f"CoinGecko error: {e}")

    # Test News API
    try:
        news_client = NewsAPIClient()
        news_data = news_client.get_top_headlines(category='business')
        print(f"Fetched {len(news_data)} news articles")
        print(news_data.head())
    except Exception as e:
        print(f"News API error: {e}")
