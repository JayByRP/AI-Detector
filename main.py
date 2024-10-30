import discord
import logging
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv
import statistics
import re
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import language_tool_python
import json
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AdvancedAIDetector:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Load pre-trained transformer model for perplexity
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                        model='distilbert-base-uncased-finetuned-sst-2-english',
                                        device=0 if torch.cuda.is_available() else -1)
        
        # Load stylometric features
        self.tfidf = TfidfVectorizer(max_features=1000)
        
        # Initialize thresholds
        self.load_thresholds()
        
    def load_thresholds(self):
        """Load or initialize detection thresholds"""
        try:
            with open('thresholds.json', 'r') as f:
                self.thresholds = json.load(f)
        except FileNotFoundError:
            self.thresholds = {
                'entropy_threshold': 4.2,
                'perplexity_threshold': 50.0,
                'coherence_threshold': 0.7,
                'repetition_threshold': 0.3,
                'grammar_threshold': 0.8,
                'sentiment_variance_threshold': 0.2
            }
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity using GPT-2 tokenizer"""
        try:
            tokens = self.tokenizer.encode(text, return_tensors='pt')
            token_probs = torch.softmax(tokens, dim=-1)
            return float(-torch.mean(torch.log(token_probs)))
        except Exception as e:
            logging.error(f"Perplexity calculation error: {e}")
            return 0.0

    def analyze_grammar(self, text: str) -> Tuple[float, List[str]]:
        """Analyze grammar and writing style"""
        matches = self.language_tool.check(text)
        error_density = len(matches) / len(text.split())
        error_types = [match.ruleId for match in matches]
        return error_density, error_types

    def calculate_coherence(self, text: str) -> float:
        """Calculate text coherence using NLP analysis"""
        doc = self.nlp(text)
        
        # Analyze sentence transitions
        coherence_scores = []
        prev_sent = None
        
        for sent in doc.sents:
            if prev_sent is not None:
                # Calculate similarity between consecutive sentences
                similarity = prev_sent.similarity(sent)
                coherence_scores.append(similarity)
            prev_sent = sent
            
        return np.mean(coherence_scores) if coherence_scores else 0.0

    def analyze_stylometric_features(self, text: str) -> Dict[str, float]:
        """Extract stylometric features from text"""
        doc = self.nlp(text)
        
        # Calculate various stylometric metrics
        word_lengths = [len(token.text) for token in doc if not token.is_punct]
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        
        # POS tag distribution
        pos_dist = Counter([token.pos_ for token in doc])
        pos_entropy = entropy(list(pos_dist.values()))
        
        # Lexical density
        content_words = len([token for token in doc if not token.is_stop])
        total_words = len([token for token in doc if not token.is_punct])
        lexical_density = content_words / total_words if total_words > 0 else 0
        
        return {
            'avg_word_length': np.mean(word_lengths),
            'word_length_variance': np.var(word_lengths),
            'avg_sentence_length': np.mean(sentence_lengths),
            'sentence_length_variance': np.var(sentence_lengths),
            'pos_entropy': pos_entropy,
            'lexical_density': lexical_density
        }

    def calculate_sentiment_variance(self, text: str) -> float:
        """Calculate variance in sentiment across sentences"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
            
        sentiments = []
        for sent in sentences:
            sentiment = self.sentiment_analyzer(sent)[0]
            sentiment_score = 1.0 if sentiment['label'] == 'POSITIVE' else 0.0
            sentiments.append(sentiment_score)
            
        return np.var(sentiments)

    def detect_repetitive_patterns(self, text: str) -> Dict[str, float]:
        """Detect repetitive patterns and phrases"""
        words = word_tokenize(text.lower())
        
        # Calculate n-gram repetition for different n
        repetition_scores = {}
        for n in range(2, 5):
            ngram_list = list(ngrams(words, n))
            if ngram_list:
                ngram_freq = Counter(ngram_list)
                max_freq = max(ngram_freq.values())
                repetition_scores[f'{n}-gram_repetition'] = max_freq / len(ngram_list)
                
        return repetition_scores

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Comprehensive text analysis using multiple detection methods"""
        try:
            # Basic text preprocessing
            text = text.strip()
            if not text:
                return {'ai_score': 0.0}

            # Calculate all metrics
            perplexity = self.calculate_perplexity(text)
            coherence = self.calculate_coherence(text)
            grammar_density, error_types = self.analyze_grammar(text)
            sentiment_variance = self.calculate_sentiment_variance(text)
            stylometric_features = self.analyze_stylometric_features(text)
            repetition_patterns = self.detect_repetitive_patterns(text)
            
            # Combine all features for final scoring
            feature_scores = {
                'perplexity_score': min(1.0, perplexity / self.thresholds['perplexity_threshold']),
                'coherence_score': coherence / self.thresholds['coherence_threshold'],
                'grammar_score': 1.0 - (grammar_density / self.thresholds['grammar_threshold']),
                'sentiment_variance_score': sentiment_variance / self.thresholds['sentiment_variance_threshold'],
                'repetition_score': max(repetition_patterns.values()) if repetition_patterns else 0.0
            }
            
            # Calculate weighted final score
            weights = {
                'perplexity_score': 0.3,
                'coherence_score': 0.2,
                'grammar_score': 0.2,
                'sentiment_variance_score': 0.15,
                'repetition_score': 0.15
            }
            
            ai_score = sum(score * weights[metric] for metric, score in feature_scores.items()) * 100
            
            # Compile detailed results
            return {
                'ai_score': ai_score,
                'perplexity': perplexity,
                'coherence': coherence,
                'grammar_density': grammar_density,
                'sentiment_variance': sentiment_variance,
                'error_types': error_types,
                'stylometric_features': stylometric_features,
                'repetition_patterns': repetition_patterns
            }
            
        except Exception as e:
            logging.error(f"Error in text analysis: {e}")
            return {'ai_score': 0.0}

class AIDetectorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        # Initialize detector
        self.detector = AdvancedAIDetector()
        
        # Load configuration
        self.load_config()
        
        # Cache for processed messages
        self.processed_messages = set()
        
    def load_config(self):
        """Load bot configuration from environment variables"""
        self.monitored_categories = set(
            int(cat_id) for cat_id in 
            os.getenv('MONITORED_CATEGORY_IDS', '').split(',') 
            if cat_id
        )
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '70'))
        self.min_chars = int(os.getenv('MIN_CHARS', '50'))
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    async def on_ready(self):
        """Log when bot is ready"""
        logger.info(f"Bot is ready! Logged in as {self.user.name}")
        logger.info(f"Monitoring categories: {self.monitored_categories}")
        logger.info(f"Alert threshold: {self.alert_threshold}%")

    async def on_message(self, message: discord.Message):
        """Process new messages"""
        try:
            if self.should_skip_message(message):
                return

            logger.info(f"Processing message from {message.author.name}")
            
            # Analyze message
            results = self.detector.analyze_text(message.content)
            score = results['ai_score']
            
            if score >= self.alert_threshold:
                await self.alert_moderators(message, results)
                self.processed_messages.add(message.id)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def should_skip_message(self, message: discord.Message) -> bool:
        """Determine if message should be skipped"""
        return (
            message.author.bot or
            not message.guild or
            not message.channel or
            not message.channel.category or
            message.channel.category.id not in self.monitored_categories or
            len(message.content) < self.min_chars or
            message.id in self.processed_messages
        )

    async def alert_moderators(self, message: discord.Message, results: Dict[str, float]):
        """Send detailed alert about potential AI content"""
        try:
            embed = self.create_alert_embed(message, results)
            
            logs_channel_id = int(os.getenv('LOGS_CHANNEL_ID'))
            logs_channel = message.guild.get_channel(logs_channel_id)
            
            if logs_channel:
                await logs_channel.send(embed=embed)
                logger.info(f"Alert sent for message {message.id}")
            else:
                logger.error("Logs channel not found")
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def create_alert_embed(self, message: discord.Message, results: Dict[str, float]) -> discord.Embed:
        """Create detailed embed for alert message"""
        embed = discord.Embed(
            title="🤖 Potential AI-Generated Content Detected",
            color=0x800000,
            timestamp=datetime.utcnow()
        )
        
        message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
        
        embed.add_field(
            name="Message Link",
            value=f"[Click to view message]({message_link})",
            inline=False
        )
        
        embed.add_field(name="AI Probability", value=f"`{results['ai_score']:.1f}%`", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        
        # Add detailed metrics
        embed.add_field(
            name="Primary Metrics", 
            value=f"Perplexity: `{results['perplexity']:.2f}`\n" \
                  f"Coherence: `{results['coherence']:.2f}`\n" \
                  f"Grammar Density: `{results['grammar_density']:.2f}`",
            inline=False
        )
        
        # Add stylometric features
        if 'stylometric_features' in results:
            style_metrics = results['stylometric_features']
            embed.add_field(
                name="Style Analysis",
                value=f"Lexical Density: `{style_metrics['lexical_density']:.2f}`\n" \
                      f"POS Entropy: `{style_metrics['pos_entropy']:.2f}`\n" \
                      f"Sentence Variance: `{style_metrics['sentence_length_variance']:.2f}`",
                inline=False
            )
        
        embed.add_field(name="Author", value=message.author.mention, inline=True)
        
        return embed

async def main():
    """Initialize and run the bot"""
    try:
        bot = AIDetectorBot()
        async with bot:
            await bot.start(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())