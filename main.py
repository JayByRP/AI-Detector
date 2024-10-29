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
        self.ignored_users = {431544605209788416, 742638883308568616}
        
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

    # [Previous analyze_message, check_writer, and check_sapling methods remain the same]

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