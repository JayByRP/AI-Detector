import discord
import os
from discord.ext import commands
import aiohttp
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
from typing import Tuple, List, Dict, Optional
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
        super().__init__(command_prefix='!', intents=intents)  # Added prefix for commands
        
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
        self.cooldowns: Dict[int, datetime] = defaultdict(datetime.now)
        self.cooldown_duration = timedelta(minutes=int(os.getenv('COOLDOWN_MINUTES', '5')))
        
        # Cache for processed messages
        self.processed_messages = set()
        self.session: Optional[aiohttp.ClientSession] = None

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        await self.load_commands()

    async def load_commands(self):
        """Load bot commands"""
        @self.command(name='setalert')
        @commands.has_permissions(administrator=True)
        async def set_alert_threshold(ctx, threshold: float):
            """Set the AI detection alert threshold"""
            if 0 <= threshold <= 100:
                self.alert_threshold = threshold
                await ctx.send(f"Alert threshold set to {threshold}%")
            else:
                await ctx.send("Threshold must be between 0 and 100")

        @self.command(name='status')
        @commands.has_permissions(administrator=True)
        async def status(ctx):
            """Check bot status and configuration"""
            embed = discord.Embed(
                title="AI Detector Status",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="Alert Threshold", value=f"{self.alert_threshold}%")
            embed.add_field(name="Monitored Categories", value=len(self.monitored_categories))
            embed.add_field(name="Writer API", value="✅" if self.writer_api_key else "❌")
            embed.add_field(name="Sapling API", value="✅" if self.sapling_api_key else "❌")
            await ctx.send(embed=embed)

    async def close(self):
        if self.session:
            await self.session.close()
        await super().close()

class AIDetectionService:
    @staticmethod
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
                timeout=10  # Added timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('ai_content_score', 0))
                logger.warning(f"Writer API returned status {response.status}")
                return 0
        except asyncio.TimeoutError:
            logger.error("Writer API timeout")
            return 0
        except Exception as e:
            logger.error(f"Writer API error: {e}")
            return 0

    @staticmethod
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
                timeout=10  # Added timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('score', 0)) * 100
                logger.warning(f"Sapling API returned status {response.status}")
                return 0
        except asyncio.TimeoutError:
            logger.error("Sapling API timeout")
            return 0
        except Exception as e:
            logger.error(f"Sapling API error: {e}")
            return 0

class MessageAnalyzer:
    @staticmethod
    async def analyze_message(message: discord.Message, bot: AIDetectorBot) -> Tuple[float, List[str]]:
        """Analyze message using multiple AI detection services"""
        if len(message.content) < bot.min_chars:
            return 0, []

        if message.id in bot.processed_messages:
            return 0, ["Message already processed"]

        # Check cooldown
        author_id = message.author.id
        if datetime.now() - bot.cooldowns[author_id] < bot.cooldown_duration:
            return 0, ["Rate limited"]
        
        analysis_details = []
        tasks = []
        service_names = []
        
        if bot.writer_api_key:
            tasks.append(AIDetectionService.check_writer(bot.session, message.content, bot.writer_api_key))
            service_names.append("Writer")
        if bot.sapling_api_key:
            tasks.append(AIDetectionService.check_sapling(bot.session, message.content, bot.sapling_api_key))
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
            return 0, ["No valid results from AI detection services"]

        # Update cooldown and processed messages
        bot.cooldowns[author_id] = datetime.now()
        bot.processed_messages.add(message.id)
        
        final_score = sum(valid_scores) / len(valid_scores)
        return final_score, analysis_details

class AIDetectionCog(commands.Cog):
    def __init__(self, bot: AIDetectorBot):
        self.bot = bot
        self.message_analyzer = MessageAnalyzer()

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

            score, analysis = await self.message_analyzer.analyze_message(message, self.bot)
            
            if score >= self.bot.alert_threshold:
                await self.alert_moderators(message, score, analysis)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    async def alert_moderators(self, message: discord.Message, score: float, analysis_details: List[str]):
        """Send alert to moderators about potential AI content"""
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
        
        # Add message content preview (truncated if too long)
        content_preview = message.content[:1000] + "..." if len(message.content) > 1000 else message.content
        embed.add_field(name="Message Content", value=content_preview, inline=False)
        
        logs_channel_id = int(os.getenv('LOGS_CHANNEL_ID'))
        logs_channel = message.guild.get_channel(logs_channel_id)
        
        if logs_channel:
            try:
                await logs_channel.send(embed=embed)
            except discord.errors.HTTPException as e:
                logger.error(f"Failed to send alert: {e}")

def main():
    """Initialize and run the bot"""
    try:
        bot = AIDetectorBot()
        bot.add_cog(AIDetectionCog(bot))
        
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            raise ValueError("No Discord token found in environment variables")
            
        bot.run(token)
        
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()