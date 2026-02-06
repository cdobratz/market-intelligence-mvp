"""
Sentiment Analysis Module - Extract sentiment features from news data

Provides functions to analyze sentiment from financial news:
- Sentiment scoring (positive, negative, neutral)
- Sentiment aggregation by time period
- Sentiment intensity analysis
- FinBERT-based transformer sentiment analysis (high accuracy)
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import transformers for FinBERT support
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. FinBERT sentiment analysis unavailable.")


class TransformerSentimentAnalyzer:
    """
    High-accuracy sentiment analyzer using FinBERT transformer model.

    FinBERT is specifically trained on financial text and provides
    15-25% accuracy improvement over keyword-based approaches.

    Features:
    - Context-aware sentiment detection
    - Handles negation properly ("not bullish" = bearish)
    - Pre-trained on financial news corpus
    - Supports batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: int = -1,  # -1 for CPU, 0+ for GPU
        batch_size: int = 32,
    ):
        """
        Initialize the transformer-based sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Device to run on (-1 for CPU, 0+ for GPU)
            batch_size: Batch size for processing multiple texts
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for TransformerSentimentAnalyzer. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._classifier = None

    @property
    def classifier(self):
        """Lazy load the classifier to avoid loading model on import."""
        if self._classifier is None:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self._classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment model loaded successfully")
        return self._classifier

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        if pd.isna(text) or not text.strip():
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral"
            }

        try:
            result = self.classifier(text[:512])[0]  # Truncate to max length
            label = result["label"].lower()
            score = result["score"]

            # Map FinBERT labels to standardized format
            if label == "positive":
                sentiment_score = score
                positive = score
                negative = 0.0
                neutral = 1.0 - score
            elif label == "negative":
                sentiment_score = -score
                positive = 0.0
                negative = score
                neutral = 1.0 - score
            else:  # neutral
                sentiment_score = 0.0
                positive = 0.0
                negative = 0.0
                neutral = score

            return {
                "sentiment": label,
                "score": score,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "sentiment_score": sentiment_score,
                "sentiment_label": label
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral"
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts efficiently using batching.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment dictionaries
        """
        results = []

        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if pd.isna(text) or not str(text).strip():
                results.append({
                    "sentiment": "neutral",
                    "score": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral"
                })
            else:
                valid_texts.append(str(text)[:512])
                valid_indices.append(i)
                results.append(None)  # Placeholder

        if not valid_texts:
            return results

        # Process in batches
        try:
            batch_results = self.classifier(valid_texts, batch_size=self.batch_size)

            for idx, result in zip(valid_indices, batch_results):
                label = result["label"].lower()
                score = result["score"]

                if label == "positive":
                    sentiment_score = score
                    positive, negative, neutral = score, 0.0, 1.0 - score
                elif label == "negative":
                    sentiment_score = -score
                    positive, negative, neutral = 0.0, score, 1.0 - score
                else:
                    sentiment_score = 0.0
                    positive, negative, neutral = 0.0, 0.0, score

                results[idx] = {
                    "sentiment": label,
                    "score": score,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": label
                }
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            # Fill remaining with neutral
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "positive": 0.0,
                        "negative": 0.0,
                        "neutral": 1.0,
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral"
                    }

        return results

    def analyze_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment of multiple texts and return DataFrame.

        Args:
            texts: List of texts to analyze

        Returns:
            DataFrame with sentiment analysis results
        """
        results = self.analyze_batch(texts)
        return pd.DataFrame(results)


class SentimentAnalyzer:
    """
    Simple sentiment analyzer using keyword-based approach
    
    For production use, consider using:
    - VADER (nltk)
    - TextBlob
    - Transformers-based models (DistilBERT)
    """
    
    # Positive keywords
    POSITIVE_KEYWORDS = {
        "bullish", "gain", "surge", "rally", "jump", "spike",
        "strong", "outperform", "upgrade", "beat", "growth",
        "profit", "success", "boom", "soar", "upside",
        "positive", "optimistic", "increase", "recover",
        "momentum", "breakthrough", "surge", "rise"
    }
    
    # Negative keywords
    NEGATIVE_KEYWORDS = {
        "bearish", "loss", "crash", "plunge", "tumble",
        "weak", "underperform", "downgrade", "miss", "decline",
        "loss", "failure", "bust", "plummet", "downside",
        "negative", "pessimistic", "decrease", "struggle",
        "collapse", "breakdown", "fall", "drop", "recession"
    }
    
    def __init__(self) -> None:
        """Initialize sentiment analyzer"""
        self.positive_count = 0
        self.negative_count = 0
        self.neutral_count = 0
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove common symbols
        text = text.replace("#", " ").replace("@", " ")
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        positive_count = sum(1 for word in words if word in self.POSITIVE_KEYWORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_KEYWORDS)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral"
            }
        
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        sentiment_score = positive_ratio - negative_ratio  # Range: -1 to 1
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": 1.0 - (positive_ratio + negative_ratio),
            "sentiment_score": sentiment_score,
            "sentiment_label": label
        }
    
    def analyze_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment analysis
        """
        results = []
        
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            results.append(sentiment)
        
        return pd.DataFrame(results)


def extract_sentiment_features(
    news_df: pd.DataFrame,
    text_column: str = "title",
    date_column: str = "publishedAt"
) -> pd.DataFrame:
    """
    Extract sentiment features from news DataFrame
    
    Args:
        news_df: News DataFrame
        text_column: Column name for text analysis
        date_column: Column name for dates
        
    Returns:
        DataFrame with sentiment features
    """
    if text_column not in news_df.columns:
        logger.warning(f"Column {text_column} not found")
        return news_df
    
    # Analyze sentiment for each article
    analyzer = SentimentAnalyzer()
    sentiment_results = analyzer.analyze_texts(news_df[text_column].fillna(""))
    
    # Combine with original data
    result = news_df.copy()
    for col in sentiment_results.columns:
        result[f"sentiment_{col}"] = sentiment_results[col]
    
    return result


def aggregate_sentiment_by_date(
    news_df: pd.DataFrame,
    date_column: str = "publishedAt",
    window: str = "D"
) -> pd.DataFrame:
    """
    Aggregate sentiment scores by time period
    
    Args:
        news_df: News DataFrame with sentiment features
        date_column: Column name for dates
        window: Resampling window ('D' for daily, 'W' for weekly)
        
    Returns:
        DataFrame with aggregated sentiment
    """
    if date_column not in news_df.columns:
        logger.warning(f"Column {date_column} not found")
        return pd.DataFrame()
    
    # Ensure date column is datetime
    news_df = news_df.copy()
    news_df[date_column] = pd.to_datetime(news_df[date_column])
    news_df.set_index(date_column, inplace=True)
    
    # Find sentiment columns
    sentiment_cols = [col for col in news_df.columns if col.startswith("sentiment_")]
    
    if not sentiment_cols:
        logger.warning("No sentiment columns found")
        return pd.DataFrame()
    
    # Aggregate by date
    aggregated = news_df[sentiment_cols].resample(window).agg({
        "sentiment_score": ["mean", "std", "min", "max", "count"],
        "sentiment_positive": ["mean"],
        "sentiment_negative": ["mean"],
        "sentiment_neutral": ["mean"],
    })
    
    # Flatten column names
    aggregated.columns = ["_".join(col).strip() for col in aggregated.columns.values]
    
    return aggregated


def get_sentiment_by_symbol(
    news_df: pd.DataFrame,
    symbol_column: str = "symbol"
) -> pd.DataFrame:
    """
    Get average sentiment by asset symbol
    
    Args:
        news_df: News DataFrame with sentiment features
        symbol_column: Column name for symbols
        
    Returns:
        DataFrame with sentiment by symbol
    """
    if symbol_column not in news_df.columns:
        logger.warning(f"Column {symbol_column} not found")
        return pd.DataFrame()
    
    sentiment_cols = [col for col in news_df.columns if col.startswith("sentiment_")]
    
    if not sentiment_cols:
        logger.warning("No sentiment columns found")
        return pd.DataFrame()
    
    result = news_df.groupby(symbol_column)[sentiment_cols].mean()
    
    return result


def calculate_sentiment_momentum(
    sentiment_series: pd.Series,
    window: int = 5
) -> pd.Series:
    """
    Calculate sentiment momentum (change in sentiment)
    
    Args:
        sentiment_series: Series of sentiment scores
        window: Window size for momentum calculation
        
    Returns:
        Series with sentiment momentum
    """
    momentum = sentiment_series.diff(window)
    return momentum


def detect_sentiment_shifts(
    sentiment_df: pd.DataFrame,
    threshold: float = 0.3,
    date_column: str = None
) -> List[Dict[str, any]]:
    """
    Detect significant shifts in sentiment
    
    Args:
        sentiment_df: DataFrame with sentiment data
        threshold: Threshold for significant change
        date_column: Optional date column
        
    Returns:
        List of detected shifts with timestamps and magnitudes
    """
    shifts = []
    
    # Find sentiment score column
    sentiment_cols = [col for col in sentiment_df.columns if "score" in col]
    
    if not sentiment_cols:
        logger.warning("No sentiment score column found")
        return shifts
    
    sentiment_col = sentiment_cols[0]
    changes = sentiment_df[sentiment_col].diff().abs()
    
    for idx, change in changes.items():
        if change > threshold:
            shift_dict = {
                "magnitude": change,
                "timestamp": idx if date_column is None else sentiment_df.loc[idx, date_column],
                "sentiment_before": sentiment_df[sentiment_col].shift(1)[idx],
                "sentiment_after": sentiment_df[sentiment_col][idx]
            }
            shifts.append(shift_dict)
    
    return shifts


class NewsProcessor:
    """
    Comprehensive news processing pipeline
    """
    
    def __init__(self) -> None:
        """Initialize news processor"""
        self.analyzer = SentimentAnalyzer()
        self.processed_count = 0
    
    def process_news_data(
        self,
        news_df: pd.DataFrame,
        text_column: str = "title",
        date_column: str = "publishedAt",
        aggregate_window: str = "D"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete news processing pipeline
        
        Args:
            news_df: Raw news data
            text_column: Text column for sentiment
            date_column: Date column
            aggregate_window: Aggregation window
            
        Returns:
            Tuple of (processed_news, aggregated_sentiment)
        """
        logger.info(f"Processing {len(news_df)} news articles...")
        
        # Extract sentiment features
        processed_news = extract_sentiment_features(
            news_df,
            text_column=text_column,
            date_column=date_column
        )
        
        # Aggregate sentiment
        aggregated_sentiment = aggregate_sentiment_by_date(
            processed_news,
            date_column=date_column,
            window=aggregate_window
        )
        
        self.processed_count += len(news_df)
        logger.info(f"Processing complete. Total articles: {self.processed_count}")
        
        return processed_news, aggregated_sentiment
    
    def get_sentiment_statistics(
        self,
        sentiment_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get summary statistics of sentiment
        
        Args:
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            Dictionary with sentiment statistics
        """
        sentiment_cols = [col for col in sentiment_df.columns if col.startswith("sentiment_")]
        
        if not sentiment_cols:
            return {}
        
        stats = {}
        
        for col in sentiment_cols:
            if col in sentiment_df.columns:
                stats[f"{col}_mean"] = sentiment_df[col].mean()
                stats[f"{col}_std"] = sentiment_df[col].std()
                stats[f"{col}_min"] = sentiment_df[col].min()
                stats[f"{col}_max"] = sentiment_df[col].max()
        
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    sample_news = pd.DataFrame({
        "title": [
            "Stock market surges as bullish sentiment takes hold",
            "Market crashes after negative earnings report",
            "Neutral trading session with mixed signals",
            "Tech stocks rally on positive growth outlook"
        ],
        "publishedAt": pd.date_range("2024-01-01", periods=4),
        "source": ["Reuters", "Reuters", "AP", "Bloomberg"]
    })
    
    # Process news
    processor = NewsProcessor()
    processed, aggregated = processor.process_news_data(sample_news)
    
    print("Processed News:")
    print(processed[["title", "sentiment_sentiment_score", "sentiment_sentiment_label"]])
    
    print("\nAggregated Sentiment:")
    print(aggregated)
    
    # Statistics
    stats = processor.get_sentiment_statistics(processed)
    print("\nSentiment Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
