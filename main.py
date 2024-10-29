import discord
import os
from discord.ext import commands
import aiohttp
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
from typing import Tuple, List, Dict
from collections import defaultdict

# Load environment variables
load_dotenv()

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

class AIDetectorBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=None, intents=intents)
        
        # Load configuration
        self.monitored_categories = set(
            int(cat_id) for cat_id in 
            os.getenv('MONITORED_CATEGORY_IDS', '').split(',') 
            if cat_id
        )
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '70'))
        self.min_chars = int(os.getenv('MIN_CHARS', '50'))
        
        # API keys
        self.writer_api_key = os.getenv('WRITER_API_KEY')
        self.sapling_api_key = os.getenv('SAPLING_API_KEY')
        
        # Rate limiting
        self.cooldowns = defaultdict(lambda: datetime.now() - timedelta(minutes=10))
        self.cooldown_duration = timedelta(minutes=int(os.getenv('COOLDOWN_MINUTES', '5')))
        
        # Cache for processed messages
        self.processed_messages = set()
        self.session = None

    async def setup_hook(self):
        """Initialize bot services"""
        logger.info("Initializing bot services...")
        self.session = aiohttp.ClientSession()
        logger.info(f"Monitoring {len(self.monitored_categories)} categories")
        logger.info(f"Alert threshold set to {self.alert_threshold}%")

    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        await super().close()

async def check_writer(session: aiohttp.ClientSession, text: str, api_key: str) -> float:
    """Check text using Writer.com's AI detection API"""
    try:
        async with session.post(
            'https://enterprise-api.writer.com/content/detect',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={'text': text},
            timeout=10
        ) as response:
            if response.status == 200:
                data = await response.json()
                score = float(data.get('ai_content_score', 0))
                logger.info(f"Writer API score: {score}")
                return score
            logger.warning(f"Writer API returned status {response.status}")
            return 0
    except Exception as e:
        logger.error(f"Writer API error: {e}")
        return 0

async def check_sapling(session: aiohttp.ClientSession, text: str, api_key: str) -> float:
    """Check text using Sapling's AI detection API"""
    try:
        async with session.post(
            'https://api.sapling.ai/api/v1/aidetect',
            headers={'Content-Type': 'application/json'},
            json={
                'key': api_key,
                'text': text
            },
            timeout=10
        ) as response:
            if response.status == 200:
                data = await response.json()
                score = float(data.get('score', 0)) * 100
                logger.info(f"Sapling API score: {score}")
                return score
            logger.warning(f"Sapling API returned status {response.status}")
            return 0
    except Exception as e:
        logger.error(f"Sapling API error: {e}")
        return 0

async def analyze_message(message: discord.Message, bot: AIDetectorBot) -> Tuple[float, List[str]]:
    """Analyze message using multiple AI detection services"""
    if len(message.content) < bot.min_chars:
        logger.debug(f"Message too short: {len(message.content)} chars")
        return 0, []

    if message.id in bot.processed_messages:
        logger.debug(f"Message {message.id} already processed")
        return 0, []

    # Check cooldown
    author_id = message.author.id
    if datetime.now() - bot.cooldowns[author_id] < bot.cooldown_duration:
        logger.debug(f"Rate limit for user {author_id}")
        return 0, []
    
    logger.info(f"Analyzing message {message.id} from user {message.author.name}")
    
    analysis_details = []
    tasks = []
    service_names = []
    
    if bot.writer_api_key:
        tasks.append(check_writer(bot.session, message.content, bot.writer_api_key))
        service_names.append("Writer")
    if bot.sapling_api_key:
        tasks.append(check_sapling(bot.session, message.content, bot.sapling_api_key))
        service_names.append("Sapling")

    if not tasks:
        logger.error("No API keys configured")
        return 0, ["No AI detection services available"]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_scores = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"{service_names[i]} error: {result}")
            continue
        if isinstance(result, (int, float)) and result > 0:
            valid_scores.append(result)
            analysis_details.append(f"{service_names[i]}: {result:.1f}%")

    if not valid_scores:
        logger.warning("No valid results from AI detection services")
        return 0, ["No valid results from AI detection services"]

    # Update cooldown and processed messages
    bot.cooldowns[author_id] = datetime.now()
    bot.processed_messages.add(message.id)
    
    final_score = sum(valid_scores) / len(valid_scores)
    logger.info(f"Final score for message {message.id}: {final_score:.1f}%")
    return final_score, analysis_details

async def alert_moderators(message: discord.Message, score: float, analysis_details: List[str]):
    """Send alert to moderators about potential AI content"""
    try:
        embed = discord.Embed(
            title="🤖 Potential AI-Generated Content Detected",
            color=discord.Color.red(),
            timestamp=datetime.utcnow()
        )
        
        message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
        
        embed.add_field(
            name="Message Link",
            value=f"[Click to view message]({message_link})",
            inline=False
        )
        
        embed.add_field(name="AI Probability", value=f"`{score:.1f}%`", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Author", value=message.author.mention, inline=True)
        
        if analysis_details:
            embed.add_field(
                name="Detection Services",
                value="\n".join(f"• {detail}" for detail in analysis_details),
                inline=False
            )
        
        content_preview = message.content[:1000] + "..." if len(message.content) > 1000 else message.content
        embed.add_field(name="Message Content", value=content_preview, inline=False)
        
        logs_channel_id = int(os.getenv('LOGS_CHANNEL_ID'))
        logs_channel = message.guild.get_channel(logs_channel_id)
        
        if logs_channel:
            await logs_channel.send(embed=embed)
            logger.info(f"Alert sent for message {message.id}")
        else:
            logger.error("Logs channel not found")
            
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

class AIDetectionCog(commands.Cog):
    def __init__(self, bot: AIDetectorBot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        """Log when bot is ready"""
        logger.info(f"Bot is ready! Logged in as {self.bot.user.name}")
        logger.info(f"Monitoring {len(self.bot.monitored_categories)} categories")
        logger.info(f"Alert threshold: {self.bot.alert_threshold}%")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Monitor messages in specified categories"""
        try:
            if (
                message.author.bot or
                not message.guild or
                not message.channel.category or
                message.channel.category.id not in self.bot.monitored_categories
            ):
                return

            score, analysis = await analyze_message(message, self.bot)
            
            if score >= self.bot.alert_threshold:
                await alert_moderators(message, score, analysis)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

async def main():
    """Initialize and run the bot"""
    try:
        bot = AIDetectorBot()
        async with bot:
            await bot.add_cog(AIDetectionCog(bot))
            await bot.start(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())