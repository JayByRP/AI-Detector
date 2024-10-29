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

class AIDetectorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        # Load configuration
        self.monitored_categories = set(
            int(cat_id) for cat_id in 
            os.getenv('MONITORED_CATEGORY_IDS', '').split(',') 
            if cat_id
        )
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '70'))
        self.min_chars = int(os.getenv('MIN_CHARS', '50'))
        
        # Ignored user IDs
        #self.ignored_users = {431544605209788416, 742638883308568616}
        self.ignored_users = {431544605209788416}
        
        # API keys
        self.writer_api_key = os.getenv('WRITER_API_KEY')
        self.sapling_api_key = os.getenv('SAPLING_API_KEY')
        
        # Rate limiting
        self.cooldowns = defaultdict(lambda: datetime.now() - timedelta(minutes=10))
        self.cooldown_duration = timedelta(minutes=int(os.getenv('COOLDOWN_MINUTES', '5')))
        
        # Cache for processed messages
        self.processed_messages = set()
        self.session = None
        
        # Start the keep-alive server
        self.bg_task = None

    async def setup_hook(self):
        """Initialize bot services"""
        logger.info("Initializing bot services...")
        self.session = aiohttp.ClientSession()
        self.bg_task = self.loop.create_task(self.keep_alive())
        logger.info(f"Monitoring categories: {self.monitored_categories}")
        logger.info(f"Alert threshold set to {self.alert_threshold}%")

    async def keep_alive(self):
        """Keep-alive server for UptimeRobot monitoring"""
        from aiohttp import web
        
        async def handle(request):
            return web.Response(text="Bot is alive!")

        app = web.Application()
        app.router.add_get("/", handle)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        logger.info("Keep-alive server started on port 8080")

    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.bg_task:
            self.bg_task.cancel()
        await super().close()

    async def on_ready(self):
        """Log when bot is ready"""
        logger.info(f"Bot is ready! Logged in as {self.user.name}")
        logger.info(f"Monitoring categories: {self.monitored_categories}")
        logger.info(f"Alert threshold: {self.alert_threshold}%")

    async def on_message(self, message: discord.Message):
        """Monitor messages in specified categories"""
        try:
            # Skip bot messages and ignored users
            if message.author.bot or message.author.id in self.ignored_users:
                return

            # Skip DMs
            if not message.guild:
                return

            # Skip if not in a channel
            if not message.channel:
                return

            # Skip if not in a category
            if not message.channel.category:
                return

            # Check if message is in a monitored category
            category_id = message.channel.category.id
            if category_id not in self.monitored_categories:
                logger.debug(f"Message in non-monitored category {category_id}")
                return

            logger.info(f"Processing message in category {category_id}")
            
            score, analysis = await self.analyze_message(message)
            
            if score >= self.alert_threshold:
                await self.alert_moderators(message, score, analysis)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    async def analyze_message(self, message: discord.Message) -> Tuple[float, List[str]]:
        """Analyze message using multiple AI detection services"""
        if len(message.content) < self.min_chars:
            logger.debug(f"Message too short: {len(message.content)} chars")
            return 0, []

        if message.id in self.processed_messages:
            logger.debug(f"Message {message.id} already processed")
            return 0, []

        # Check cooldown
        author_id = message.author.id
        if datetime.now() - self.cooldowns[author_id] < self.cooldown_duration:
            logger.debug(f"Rate limit for user {author_id}")
            return 0, []
        
        logger.info(f"Analyzing message {message.id} from user {message.author.name}")
        
        analysis_details = []
        tasks = []
        service_names = []
        
        if self.writer_api_key:
            tasks.append(self.check_writer(message.content))
            service_names.append("Writer")
        if self.sapling_api_key:
            tasks.append(self.check_sapling(message.content))
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
        self.cooldowns[author_id] = datetime.now()
        self.processed_messages.add(message.id)
        
        final_score = sum(valid_scores) / len(valid_scores)
        logger.info(f"Final score for message {message.id}: {final_score:.1f}%")
        return final_score, analysis_details

    async def check_writer(self, text: str) -> float:
        """Check text using Writer.com's AI detection API"""
        try:
            async with self.session.post(
                'https://enterprise-api.writer.com/content/detect',
                headers={
                    'Authorization': f'Bearer {self.writer_api_key}',
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

    async def check_sapling(self, text: str) -> float:
        """Check text using Sapling's AI detection API"""
        try:
            async with self.session.post(
                'https://api.sapling.ai/api/v1/aidetect',
                headers={'Content-Type': 'application/json'},
                json={
                    'key': self.sapling_api_key,
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

    async def alert_moderators(self, message: discord.Message, score: float, analysis_details: List[str]):
        """Send alert to moderators about potential AI content"""
        try:
            embed = discord.Embed(
                title="🤖 Potential AI-Generated Content Detected",
                color=0x800000,  # Dark red/burgundy color
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
            
            logs_channel_id = int(os.getenv('LOGS_CHANNEL_ID'))
            logs_channel = message.guild.get_channel(logs_channel_id)
            
            if logs_channel:
                await logs_channel.send(embed=embed)
                logger.info(f"Alert sent for message {message.id}")
            else:
                logger.error("Logs channel not found")
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

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