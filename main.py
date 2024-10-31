import discord
import logging
from datetime import datetime
import asyncio
import sys
import os
from dotenv import load_dotenv
import time
import re
import numpy as np
from typing import Dict, List
import spacy
from collections import Counter
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add health check routes
async def health_check(request):
    """Simple health check endpoint"""
    return web.Response(text="OK", status=200)

# Create web app for health checks
app = web.Application()
app.router.add_get('/health', health_check)

class LightweightNarrativeDetector:
    def __init__(self):
        try:
            # Use efficient spaCy model
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer')
            logger.info("Loaded spaCy model successfully")
            
            # Initialize patterns and thresholds
            self.narrative_patterns = self.load_narrative_patterns()
            self.load_thresholds()
            
            # Pre-compile regex patterns
            self.compiled_patterns = self.compile_patterns()
            
            logger.info("Lightweight Narrative Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise

    def compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Pre-compile regex patterns for efficiency"""
        compiled = {}
        for category, patterns in self.narrative_patterns.items():
            compiled[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled

    def load_narrative_patterns(self) -> Dict[str, List[str]]:
        """Load patterns common in AI-generated creative writing"""
        return {
            'generic_descriptions': [
                r'\b(suddenly|slowly|quickly|immediately)\b',
                r'\b(beautiful|handsome|gorgeous|stunning)\b',
                r'\b(smiled|grinned|smirked|frowned)\b'
            ],
            'stock_phrases': [
                r'\b(without\sa\sword|in\sthe\sblink\sof\san\seye)\b',
                r'\b(heart\sskipped\sa\sbeat|breath\scaught)\b'
            ],
            'dialogue_markers': [
                r'\".*?\"\s*(?:he|she|they)\s*(?:said|replied|asked)',
                r'\b(?:exclaimed|declared|proclaimed)\b'
            ],
            'ai_indicators': [
                r'\b(delve|testament|dynamic|moreover|tapestry|realm)\b',
                r'\b(it\sis\sworth\snoting|notably|transformative)\b'
            ]
        }

    def load_thresholds(self):
        """Initialize detection thresholds with more balanced values"""
        self.thresholds = {
            'pattern_density': 0.05,  # Reduced from 0.1
            'repetition_threshold': 0.25,  # Increased from 0.15
            'sentence_complexity': 0.3,  # Reduced from 0.4
            'dialogue_ratio': 0.1,  # Decreased from 0.3
            'min_length': 50,  # Minimum text length for reliable analysis
            'max_pattern_score': 0.7  # Cap on pattern matching contribution
        }

    def calculate_text_statistics(self, text: str) -> Dict[str, float]:
        """Calculate basic text statistics with improved metrics"""
        # Basic tokenization
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        
        # Word frequency analysis
        word_freq = Counter(words)
        unique_ratio = len(word_freq) / len(words) if words else 0
        
        # Sentence analysis
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        avg_sent_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Calculate vocabulary richness
        rare_words = sum(1 for word, count in word_freq.items() if count == 1)
        vocab_richness = rare_words / len(words) if words else 0
        
        return {
            'unique_word_ratio': unique_ratio,
            'avg_sentence_length': avg_sent_length,
            'word_count': len(words),
            'vocab_richness': vocab_richness
        }

    def detect_patterns(self, text: str) -> Dict[str, float]:
        """Detect AI patterns with improved scoring"""
        pattern_matches = {}
        text_length = len(text.split())
        
        if text_length < self.thresholds['min_length']:
            return {category: 0.0 for category in self.compiled_patterns}
        
        for category, patterns in self.compiled_patterns.items():
            matches = sum(len(pattern.findall(text)) for pattern in patterns)
            # Apply diminishing returns to pattern matches
            score = min(matches / text_length, self.thresholds['max_pattern_score'])
            pattern_matches[category] = score
            
        return pattern_matches

    def analyze_style_markers(self, text: str) -> Dict[str, float]:
        """Analyze writing style markers with more nuanced metrics"""
        # Dialogue analysis
        dialogue_count = len(re.findall(r'\".*?\"', text))
        sentences = [s for s in text.split('.') if s.strip()]
        dialogue_ratio = dialogue_count / len(sentences) if sentences else 0
        
        # Adjective and adverb density
        doc = self.nlp(text)
        adj_adv_count = sum(1 for token in doc if token.pos_ in {'ADJ', 'ADV'})
        word_count = sum(1 for token in doc if not token.is_punct and not token.is_space)
        
        # Calculate normalized densities
        style_density = adj_adv_count / word_count if word_count > 0 else 0
        
        return {
            'dialogue_ratio': min(dialogue_ratio, self.thresholds['dialogue_ratio']),
            'style_density': style_density
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Improved text analysis with balanced scoring"""
        try:
            text = text.strip()
            if not text or len(text) < self.thresholds['min_length']:
                return {'ai_score': 0.0, 'confidence': 0.0}

            # Get basic statistics
            stats = self.calculate_text_statistics(text)
            
            # Pattern detection
            patterns = self.detect_patterns(text)
            
            # Style analysis
            style = self.analyze_style_markers(text)
            
            # Calculate component scores with adjusted weights
            pattern_score = sum(patterns.values()) / len(patterns) if patterns else 0
            style_score = (style['dialogue_ratio'] + style['style_density']) / 2
            
            # Vocabulary richness reduces AI probability
            vocab_bonus = min(stats['vocab_richness'] * 0.5, 0.3)
            
            # Calculate confidence based on text length
            confidence = min(stats['word_count'] / 500, 1.0)
            
            # Weighted scoring with more balanced weights
            weights = {
                'patterns': 0.3,  # Reduced from 0.4
                'style': 0.2,     # Reduced from 0.3
                'uniqueness': 0.3,
                'vocab': 0.2      # New component
            }
            
            ai_score = (
                (pattern_score * weights['patterns']) +
                (style_score * weights['style']) +
                ((1 - stats['unique_word_ratio']) * weights['uniqueness']) -
                (vocab_bonus * weights['vocab'])
            ) * 100

            # Ensure score stays within 0-100 range
            ai_score = max(0, min(100, ai_score))

            return {
                'ai_score': ai_score,
                'confidence': confidence,
                'statistics': stats,
                'pattern_analysis': patterns,
                'style_markers': style
            }

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'ai_score': 0.0, 'confidence': 0.0}

class AIDetectorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.detector = LightweightNarrativeDetector()
        self.load_config()
        self.processed_messages = set()
        self.last_heartbeat = time.time()
        logger.info("Bot initialized successfully")

    def load_config(self):
        """Load bot configuration"""
        load_dotenv()
        
        # Load and validate monitored categories
        category_ids = os.getenv('MONITORED_CATEGORY_IDS', '')
        self.monitored_categories = set()
        if category_ids:
            try:
                self.monitored_categories = set(int(cat_id) for cat_id in category_ids.split(',') if cat_id.strip())
                logger.info(f"Monitoring categories: {self.monitored_categories}")
            except ValueError as e:
                logger.error(f"Invalid category IDs in config: {e}")
        else:
            logger.warning("No category IDs configured for monitoring")

        # Load other settings
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '70'))
        self.min_chars = int(os.getenv('MIN_CHARS', '100'))
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '300'))
        
        # Validate logs channel
        self.logs_channel_id = os.getenv('LOGS_CHANNEL_ID')
        if not self.logs_channel_id:
            logger.error("LOGS_CHANNEL_ID not configured!")
        else:
            try:
                self.logs_channel_id = int(self.logs_channel_id)
                logger.info(f"Logs will be sent to channel: {self.logs_channel_id}")
            except ValueError:
                logger.error("Invalid LOGS_CHANNEL_ID format")

        logger.info(f"Configuration loaded - Alert threshold: {self.alert_threshold}, Min chars: {self.min_chars}")

    async def on_ready(self):
        """Bot ready handler"""
        logger.info(f"Bot is ready! Logged in as {self.user.name}")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        
        # Validate channels on startup
        for guild in self.guilds:
            logs_channel = guild.get_channel(self.logs_channel_id)
            if logs_channel:
                logger.info(f"Found logs channel in guild {guild.name}")
            else:
                logger.error(f"Logs channel not found in guild {guild.name}")
            
            # Log monitored categories found
            found_categories = [cat for cat in guild.categories if cat.id in self.monitored_categories]
            logger.info(f"Found {len(found_categories)} monitored categories in {guild.name}")
            
        self.loop.create_task(self.heartbeat())

    def should_skip_message(self, message: discord.Message) -> bool:
        """Message filter with detailed logging"""
        if message.author.bot:
            logger.debug(f"Skipping bot message from {message.author.name}")
            return True
            
        if not message.guild:
            logger.debug("Skipping DM message")
            return True
            
        if not message.channel or not message.channel.category:
            logger.debug(f"Skipping message from channel without category")
            return True
            
        if message.channel.category.id not in self.monitored_categories:
            logger.debug(f"Skipping message from unmonitored category {message.channel.category.name}")
            return True
            
        if len(message.content) < self.min_chars:
            logger.debug(f"Skipping message: too short ({len(message.content)} chars)")
            return True
            
        if message.id in self.processed_messages:
            logger.debug(f"Skipping already processed message {message.id}")
            return True
            
        logger.info(f"Processing message {message.id} from {message.author.name} in {message.channel.name}")
        return False

    async def on_message(self, message: discord.Message):
        """Message handler with improved logging"""
        try:
            logger.debug(f"Received message: {message.id} in channel {message.channel.id if message.channel else 'N/A'}")
            
            if self.should_skip_message(message):
                return

            logger.info(f"Analyzing message {message.id} content length: {len(message.content)}")
            results = self.detector.analyze_text(message.content)
            score = results.get('ai_score', 0)
            
            logger.info(f"Analysis complete for message {message.id} - Score: {score:.2f}")
            
            if score >= self.alert_threshold:
                logger.info(f"High AI score detected ({score:.2f}%) - sending alert")
                await self.alert_moderators(message, results)
                self.processed_messages.add(message.id)
                logger.info(f"Alert sent and message {message.id} marked as processed")
            
        except Exception as e:
            logger.error(f"Error processing message {message.id if message else 'N/A'}: {e}", exc_info=True)

    async def alert_moderators(self, message: discord.Message, results: Dict[str, float]):
        """Send alerts with better error handling"""
        try:
            embed = self.create_alert_embed(message, results)
            logs_channel = message.guild.get_channel(self.logs_channel_id)
            
            if logs_channel:
                logger.info(f"Sending alert to logs channel {logs_channel.name}")
                await logs_channel.send(embed=embed)
                logger.info(f"Alert sent successfully for message {message.id}")
            else:
                logger.error(f"Logs channel {self.logs_channel_id} not found in guild {message.guild.name}")
                
        except discord.Forbidden:
            logger.error(f"Bot lacks permission to send messages in logs channel")
        except discord.HTTPException as e:
            logger.error(f"Failed to send alert due to Discord API error: {e}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}", exc_info=True)

    def create_alert_embed(self, message: discord.Message, results: Dict[str, float]) -> discord.Embed:
        """Create alert embed with additional information"""
        try:
            embed = discord.Embed(
                title="🤖 Potential AI-Generated Content Detected",
                color=0x800000,
                timestamp=datetime.utcnow()
            )
            
            message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
            
            embed.add_field(name="Message Link", value=f"[Click to view]({message_link})", inline=False)
            embed.add_field(name="AI Probability", value=f"`{results['ai_score']:.1f}%`", inline=True)
            embed.add_field(name="Confidence", value=f"`{results.get('confidence', 0):.1f}%`", inline=True)
            embed.add_field(name="Channel", value=message.channel.mention, inline=True)
            
            if 'statistics' in results:
                stats = results['statistics']
                stats_text = (f"Word Count: `{stats['word_count']}`\n"
                             f"Unique Words: `{stats['unique_word_ratio']:.2f}`\n"
                             f"Avg Sentence Length: `{stats['avg_sentence_length']:.1f}`\n"
                             f"Vocabulary Richness: `{stats['vocab_richness']:.2f}`")
                embed.add_field(name="Text Statistics", value=stats_text, inline=False)
            
            if 'style_markers' in results:
                style = results['style_markers']
                style_text = (f"Dialogue Ratio: `{style['dialogue_ratio']:.2f}`\n"
                             f"Style Density: `{style['style_density']:.2f}`")
                embed.add_field(name="Style Analysis", value=style_text, inline=False)
            
            embed.add_field(name="Author", value=message.author.mention, inline=True)
            
            return embed
        except Exception as e:
            logger.error(f"Error creating embed: {e}", exc_info=True)
            raise

    async def on_message(self, message: discord.Message):
        """Message handler"""
        try:
            if self.should_skip_message(message):
                return

            results = self.detector.analyze_text(message.content)
            score = results['ai_score']
            logger.info("AI score: {}".format(score))
            
            if score >= self.alert_threshold:
                await self.alert_moderators(message, results)
                self.processed_messages.add(message.id)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def should_skip_message(self, message: discord.Message) -> bool:
        """Message filter"""
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
        """Send alerts"""
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

async def start_webserver():
    """Start the webserver"""
    port = int(os.getenv('PORT', 10000))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health check webserver started on port {port}")
    return site

async def main():
    """Main entry point"""
    try:
        site = await start_webserver()
        bot = AIDetectorBot()
        async with bot:
            await bot.start(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise
    finally:
        if 'site' in locals():
            await site.stop()

if __name__ == "__main__":
    time.sleep(5)  # Startup delay
    asyncio.run(main())