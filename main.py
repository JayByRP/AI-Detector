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

    def compile_patterns(self):
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
        """Initialize detection thresholds"""
        self.thresholds = {
            'pattern_density': 0.1,
            'repetition_threshold': 0.15,
            'sentence_complexity': 0.4,
            'dialogue_ratio': 0.3
        }

    def calculate_text_statistics(self, text: str) -> Dict[str, float]:
        """Calculate basic text statistics"""
        # Basic tokenization
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc if not token.is_punct]
        
        # Word frequency analysis
        word_freq = Counter(words)
        unique_ratio = len(word_freq) / len(words) if words else 0
        
        # Sentence analysis
        sentences = [sent.text for sent in doc.sents]
        avg_sent_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        return {
            'unique_word_ratio': unique_ratio,
            'avg_sentence_length': avg_sent_length,
            'word_count': len(words)
        }

    def detect_patterns(self, text: str) -> Dict[str, float]:
        """Detect AI patterns efficiently"""
        pattern_matches = {}
        text_length = len(text.split())
        
        for category, patterns in self.compiled_patterns.items():
            matches = sum(len(pattern.findall(text)) for pattern in patterns)
            pattern_matches[category] = matches / text_length if text_length > 0 else 0
            
        return pattern_matches

    def analyze_style_markers(self, text: str) -> Dict[str, float]:
        """Analyze writing style markers"""
        # Dialogue analysis
        dialogue_count = len(re.findall(r'\".*?\"', text))
        sentences = text.split('.')
        dialogue_ratio = dialogue_count / len(sentences) if sentences else 0
        
        # Adjective density
        doc = self.nlp(text)
        adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
        adj_density = adj_count / len(doc) if len(doc) > 0 else 0
        
        return {
            'dialogue_ratio': dialogue_ratio,
            'adjective_density': adj_density
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Efficient text analysis"""
        try:
            text = text.strip()
            if not text:
                return {'ai_score': 0.0}

            # Get basic statistics
            stats = self.calculate_text_statistics(text)
            
            # Pattern detection
            patterns = self.detect_patterns(text)
            
            # Style analysis
            style = self.analyze_style_markers(text)
            
            # Calculate AI score
            pattern_score = sum(patterns.values()) / len(patterns) if patterns else 0
            style_score = (style['dialogue_ratio'] + style['adjective_density']) / 2
            uniqueness_score = stats['unique_word_ratio']
            
            # Weighted scoring
            weights = {
                'patterns': 0.4,
                'style': 0.3,
                'uniqueness': 0.3
            }
            
            ai_score = (
                pattern_score * weights['patterns'] +
                style_score * weights['style'] +
                (1 - uniqueness_score) * weights['uniqueness']
            ) * 100

            return {
                'ai_score': ai_score,
                'statistics': stats,
                'pattern_analysis': patterns,
                'style_markers': style
            }

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'ai_score': 0.0}

class AIDetectorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.detector = LightweightNarrativeDetector()
        self.load_config()
        self.processed_messages = set()
        self.last_heartbeat = time.time()
        
    def load_config(self):
        """Load bot configuration"""
        load_dotenv()
        self.monitored_categories = set(
            int(cat_id) for cat_id in 
            os.getenv('MONITORED_CATEGORY_IDS', '').split(',') 
            if cat_id
        )
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '70'))
        self.min_chars = int(os.getenv('MIN_CHARS', '100'))
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '300'))

    async def heartbeat(self):
        """Maintain bot uptime"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat >= self.heartbeat_interval:
                    logger.info("Heartbeat: Bot is alive")
                    self.last_heartbeat = current_time
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)

    async def on_ready(self):
        """Bot ready handler"""
        logger.info(f"Narrative AI Detection Bot is ready! Logged in as {self.user.name}")
        self.loop.create_task(self.heartbeat())

    def create_alert_embed(self, message: discord.Message, results: Dict[str, float]) -> discord.Embed:
        """Create alert embed"""
        embed = discord.Embed(
            title="🤖 Potential AI-Generated Content Detected",
            color=0x800000,
            timestamp=datetime.utcnow()
        )
        
        message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
        
        embed.add_field(name="Message Link", value=f"[Click to view]({message_link})", inline=False)
        embed.add_field(name="AI Probability", value=f"`{results['ai_score']:.1f}%`", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        
        if 'statistics' in results:
            stats = results['statistics']
            stats_text = (f"Word Count: `{stats['word_count']}`\n"
                         f"Unique Words: `{stats['unique_word_ratio']:.2f}`\n"
                         f"Avg Sentence Length: `{stats['avg_sentence_length']:.1f}`")
            embed.add_field(name="Text Statistics", value=stats_text, inline=False)
        
        if 'style_markers' in results:
            style = results['style_markers']
            style_text = (f"Dialogue Ratio: `{style['dialogue_ratio']:.2f}`\n"
                         f"Adjective Density: `{style['adjective_density']:.2f}`")
            embed.add_field(name="Style Analysis", value=style_text, inline=False)
        
        embed.add_field(name="Author", value=message.author.mention, inline=True)
        
        return embed

    async def on_message(self, message: discord.Message):
        """Message handler"""
        try:
            if self.should_skip_message(message):
                return

            results = self.detector.analyze_text(message.content)
            score = results['ai_score']
            
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