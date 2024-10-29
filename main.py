import discord
import os
from discord.ext import commands
import aiohttp
import logging
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from typing import Tuple, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIDetectorBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=None, intents=intents)
        
        # Load monitored categories
        self.monitored_categories = set(
            int(cat_id) for cat_id in 
            os.getenv('MONITORED_CATEGORY_IDS', '').split(',') 
            if cat_id
        )
        
        # Initialize API keys
        self.writer_api_key = os.getenv('WRITER_API_KEY')
        self.originator_api_key = os.getenv('ORIGINATOR_API_KEY')
        self.gptzero_api_key = os.getenv('GPTZERO_API_KEY')
        self.sapling_api_key = os.getenv('SAPLING_API_KEY')
        
        # Initialize session
        self.session = None

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
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
            json={'text': text}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get('ai_content_score', 0))
            return 0
    except Exception as e:
        logger.error(f"Writer API error: {e}")
        return 0

async def check_gptzero(session: aiohttp.ClientSession, text: str, api_key: str) -> float:
    """Check text using GPTZero's API"""
    try:
        async with session.post(
            'https://api.gptzero.me/v2/predict/text',
            headers={
                'X-Api-Key': api_key,
                'Content-Type': 'application/json'
            },
            json={'document': text}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get('documents', [{}])[0].get('average_generated_prob', 0)) * 100
            return 0
    except Exception as e:
        logger.error(f"GPTZero API error: {e}")
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
            }
        ) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get('score', 0)) * 100
            return 0
    except Exception as e:
        logger.error(f"Sapling API error: {e}")
        return 0

async def check_originator(session: aiohttp.ClientSession, text: str, api_key: str) -> float:
    """Check text using Originator's AI detection API"""
    try:
        async with session.post(
            'https://api.originator.ai/v1/detect',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={'text': text}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get('ai_probability', 0)) * 100
            return 0
    except Exception as e:
        logger.error(f"Originator API error: {e}")
        return 0

async def analyze_message(message: discord.Message, bot: AIDetectorBot) -> Tuple[float, List[str]]:
    """Analyze message using multiple AI detection services"""
    if len(message.content) < 50:  # Ignore very short messages
        return 0, []

    analysis_details = []
    
    # Run all API checks concurrently
    tasks = []
    service_names = []
    
    if bot.writer_api_key:
        tasks.append(check_writer(bot.session, message.content, bot.writer_api_key))
        service_names.append("Writer")
    if bot.gptzero_api_key:
        tasks.append(check_gptzero(bot.session, message.content, bot.gptzero_api_key))
        service_names.append("GPTZero")
    if bot.sapling_api_key:
        tasks.append(check_sapling(bot.session, message.content, bot.sapling_api_key))
        service_names.append("Sapling")
    if bot.originator_api_key:
        tasks.append(check_originator(bot.session, message.content, bot.originator_api_key))
        service_names.append("Originator")

    if not tasks:
        logger.error("No API keys configured")
        return 0, ["No AI detection services available"]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    valid_scores = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue
        if isinstance(result, (int, float)) and result > 0:
            valid_scores.append(result)
            analysis_details.append(f"{service_names[i]}: {result:.1f}%")

    if not valid_scores:
        return 0, ["No valid results from AI detection services"]

    # Calculate weighted average score
    final_score = sum(valid_scores) / len(valid_scores)
    
    return final_score, analysis_details

async def alert_moderators(message: discord.Message, score: float, analysis_details: List[str]):
    """Send alert to moderators about potential AI content"""
    embed = discord.Embed(
        title="🤖 Potential AI-Generated Content Detected",
        color=discord.Color.red(),
        timestamp=datetime.utcnow()
    )
    
    # Create message link
    message_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
    
    embed.add_field(
        name="Message Link",
        value=f"[Click to view message]({message_link})",
        inline=False
    )
    
    embed.add_field(
        name="AI Probability",
        value=f"`{score:.1f}%`",
        inline=True
    )
    
    embed.add_field(
        name="Channel",
        value=f"{message.channel.mention}",
        inline=True
    )
    
    embed.add_field(
        name="Author",
        value=f"{message.author.mention}",
        inline=True
    )
    
    if analysis_details:
        embed.add_field(
            name="Detection Services",
            value="\n".join(f"• {detail}" for detail in analysis_details),
            inline=False
        )
    
    # Send alert to the logs channel
    logs_channel_id = int(os.getenv('LOGS_CHANNEL_ID'))
    logs_channel = message.guild.get_channel(logs_channel_id)
    
    if logs_channel:
        try:
            await logs_channel.send(embed=embed)
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to send alert: {e}")

class AIDetectionCog(commands.Cog):
    def __init__(self, bot: AIDetectorBot):
        self.bot = bot

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
            
            if score >= 70:  # Alert threshold
                await alert_moderators(message, score, analysis)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

def main():
    """Initialize and run the bot"""
    try:
        bot = AIDetectorBot()
        bot.add_cog(AIDetectionCog(bot))
        
        # Run the bot
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            raise ValueError("No Discord token found in environment variables")
            
        bot.run(token)
        
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()