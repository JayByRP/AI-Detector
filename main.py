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
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
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

class NarrativeAIDetector:
    def __init__(self):
        try:
            # Use smaller model and disable GPU if memory is limited
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded spaCy model successfully")
            
            # Use smaller model for sentiment analysis
            self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                            model='distilbert-base-uncased-finetuned-sst-2-english',
                                            device=-1)  # Force CPU usage
            
            # Minimize NLTK downloads
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize creative writing patterns
            self.narrative_patterns = self.load_narrative_patterns()
            self.load_thresholds()
            
            logger.info("Narrative AI Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Detector: {e}")
            raise

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
                r'\b(heart\sskipped\sa\sbeat|breath\scaught\sin\s(?:his|her|their)\sthroat)\b',
                r'\b(in\sthe\scontext\sof\b)'
            ],
            'dialogue_markers': [
                r'\".*?\"\s*(?:he|she|they)\s*(?:said|replied|answered|asked)',
                r'\b(?:exclaimed|declared|proclaimed|announced)\b'
            ],
            'repetitive_patterns': [
                r'(\b\w+\b)(\s+\1\b){2,}',
                r'\b(felt|experienced|sensed)\b.*?\b(emotion|feeling)\b'
            ],
            'overused_transitions': [
                r'\b(meanwhile|however|moreover|furthermore)\b',
                r'\b(in\sthat\smoment|at\sthat\stime|just\sthen)\b'
            ],
            'descriptive_adjectives_and_verbs': [
                r'\b(delve|testament|dynamic|important\sto\sconsider|dive\sinto|moreover|tapestry|'
                r'additionally|realm|remember\sthat|vibrant|vital|arguably|certainly|elevate|'
                r'explore|in\ssummary|it\sis\sworth\snoting|notably|transformative|accordingly|'
                r'commendable|comprehensive|embrace|tantalizing|cocooned|murmur)\b'
            ]
        }

    def load_thresholds(self):
        """Initialize detection thresholds for narrative content"""
        self.thresholds = {
            'entropy_threshold': 4.5,           # Higher for creative writing
            'perplexity_threshold': 60.0,       # Adjusted for narrative complexity
            'coherence_threshold': 0.65,        # Slightly lower for creative flow
            'pattern_threshold': 0.4,           # Adjusted for narrative style
            'dialogue_ratio_threshold': 0.3,    # Expected dialogue presence
            'description_density_threshold': 0.25
        }

    def analyze_narrative_style(self, text: str) -> Dict[str, float]:
        """Analyze narrative writing style markers"""
        doc = self.nlp(text)
        
        # Analyze dialogue patterns
        dialogue_count = len(re.findall(r'\".*?\"', text))
        dialogue_ratio = dialogue_count / len(sent_tokenize(text)) if len(sent_tokenize(text)) > 0 else 0
        
        # Analyze descriptive language
        descriptive_words = len([token for token in doc 
                               if token.pos_ in ['ADJ', 'ADV']])
        description_density = descriptive_words / len(doc) if len(doc) > 0 else 0
        
        # Analyze character references
        character_mentions = len([ent for ent in doc.ents 
                                if ent.label_ == 'PERSON'])
        
        return {
            'dialogue_ratio': dialogue_ratio,
            'description_density': description_density,
            'character_count': character_mentions
        }

    def detect_ai_patterns(self, text: str) -> Dict[str, float]:
        """Detect AI patterns in narrative writing"""
        pattern_scores = {}
        
        for category, patterns in self.narrative_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            pattern_scores[category] = matches / len(text.split()) if len(text.split()) > 0 else 0
            
        return pattern_scores

    def calculate_creativity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate metrics specific to creative writing"""
        words = word_tokenize(text.lower())
        
        # Vocabulary richness
        unique_words = len(set(words)) / len(words) if len(words) > 0 else 0
        
        # Sentence variety
        sentences = sent_tokenize(text)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        sentence_variety = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Emotional range
        emotion_scores = []
        for sent in sentences:
            try:
                sentiment = self.sentiment_analyzer(sent)[0]
                emotion_scores.append(1.0 if sentiment['label'] == 'POSITIVE' else 0.0)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                continue
        
        emotional_range = np.std(emotion_scores) if emotion_scores else 0
        
        return {
            'vocabulary_richness': unique_words,
            'sentence_variety': sentence_variety,
            'emotional_range': emotional_range
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Comprehensive narrative text analysis"""
        try:
            text = text.strip()
            if not text:
                return {'ai_score': 0.0}

            # Narrative-specific analysis
            narrative_style = self.analyze_narrative_style(text)
            pattern_scores = self.detect_ai_patterns(text)
            creativity_metrics = self.calculate_creativity_metrics(text)
            
            # Calculate feature scores
            feature_scores = {
                'dialogue_score': min(1.0, narrative_style['dialogue_ratio'] / self.thresholds['dialogue_ratio_threshold']),
                'description_score': min(1.0, narrative_style['description_density'] / self.thresholds['description_density_threshold']),
                'pattern_score': min(1.0, max(pattern_scores.values()) / self.thresholds['pattern_threshold']),
                'creativity_score': min(1.0, creativity_metrics['vocabulary_richness'])
            }

            # Weighted scoring for narrative content
            weights = {
                'dialogue_score': 0.25,
                'description_score': 0.25,
                'pattern_score': 0.3,
                'creativity_score': 0.2
            }

            ai_score = sum(score * weights[metric] for metric, score in feature_scores.items()) * 100

            return {
                'ai_score': ai_score,
                'narrative_style': narrative_style,
                'pattern_analysis': pattern_scores,
                'creativity_metrics': creativity_metrics,
                'feature_scores': feature_scores
            }

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'ai_score': 0.0}

class AIDetectorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.detector = NarrativeAIDetector()
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
        self.min_chars = int(os.getenv('MIN_CHARS', '100'))  # Increased for narrative content
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
        """Create detailed alert embed for narrative content"""
        embed = discord.Embed(
            title="🤖 Potential AI-Generated Narrative Detected",
            color=0x800000,
            timestamp=datetime.utcnow()
        )
        
        message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
        
        embed.add_field(name="Message Link", value=f"[Click to view]({message_link})", inline=False)
        embed.add_field(name="AI Probability", value=f"`{results['ai_score']:.1f}%`", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        
        # Add narrative-specific metrics
        if 'narrative_style' in results:
            style = results['narrative_style']
            style_text = (f"Dialogue Ratio: `{style['dialogue_ratio']:.2f}`\n"
                         f"Description Density: `{style['description_density']:.2f}`\n"
                         f"Character References: `{style['character_count']}`")
            embed.add_field(name="Writing Style", value=style_text, inline=False)
        
        if 'creativity_metrics' in results:
            creativity = results['creativity_metrics']
            creativity_text = (f"Vocabulary Richness: `{creativity['vocabulary_richness']:.2f}`\n"
                             f"Sentence Variety: `{creativity['sentence_variety']:.2f}`\n"
                             f"Emotional Range: `{creativity['emotional_range']:.2f}`")
            embed.add_field(name="Creativity Analysis", value=creativity_text, inline=False)
        
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
        """Message filter for narrative content"""
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
        """Send alerts for detected AI content"""
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

# Modify the start_webserver function
async def start_webserver():
    """Start the webserver for health checks"""
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
        # Start the webserver
        site = await start_webserver()
        
        # Start the bot
        bot = AIDetectorBot()
        async with bot:
            await bot.start(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise
    finally:
        # Ensure the webserver is closed
        if 'site' in locals():
            await site.stop()

if __name__ == "__main__":
    time.sleep(5)  # Startup delay
    asyncio.run(main())