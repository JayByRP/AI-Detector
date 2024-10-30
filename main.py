import discord
import logging
from datetime import datetime
import asyncio
from typing import Tuple, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dotenv import load_dotenv
import os

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

class AIDetectionModel(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)  # BERT base hidden size is 768
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

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
        
        # Initialize AI detection model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AIDetectionModel()
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Cache for processed messages
        self.processed_messages = set()
        
    async def analyze_text(self, text: str) -> float:
        """Analyze text using the custom AI detection model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.item() * 100  # Convert to percentage
                
            return score
            
        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            return 0

    async def on_ready(self):
        """Log when bot is ready"""
        logger.info(f"Bot is ready! Logged in as {self.user.name}")
        logger.info(f"Monitoring categories: {self.monitored_categories}")
        logger.info(f"Alert threshold: {self.alert_threshold}%")

    async def on_message(self, message: discord.Message):
        """Monitor messages in specified categories"""
        try:
            # Skip bot messages
            if message.author.bot:
                return

            # Skip if not in a monitored category
            if not message.guild or not message.channel or not message.channel.category:
                return
                
            category_id = message.channel.category.id
            if category_id not in self.monitored_categories:
                return

            # Skip if message too short
            if len(message.content) < self.min_chars:
                return

            # Skip if already processed
            if message.id in self.processed_messages:
                return

            logger.info(f"Processing message from {message.author.name}")
            
            # Analyze message
            score = await self.analyze_text(message.content)
            
            if score >= self.alert_threshold:
                await self.alert_moderators(message, score)
                self.processed_messages.add(message.id)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    async def alert_moderators(self, message: discord.Message, score: float):
        """Send alert to moderators about potential AI content"""
        try:
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