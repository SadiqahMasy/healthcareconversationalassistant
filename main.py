import logging
from openai import OpenAI
import discord
from discord.ext import commands
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

from doctor_conversation import doctor_conversation
from patient_conversation import patientConversationAskQuestion

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True 
intents.voice_states = True
bot = commands.Bot(command_prefix='!', intents=intents)

openai_client = OpenAI(api_key=os.getenv("open-ai-api-key"))

# For simplicity we'll store our app data in-memory with the following data structure.
# onboarding_tutorials_sent = {"channel": {"user_id": OnboardingTutorial}}

doctor_channel = "dr-smith"

active_channels = {
    "elijah-carter-patient-01": {
        "patient_context": {
            "patient_info": {
                "name": "elijah carter",
                "age": 45,
                "gender": "male"
            },
            "medical_history": [
                "Hypertension",
                "Type 2 Diabetes",
                "Previous knee surgery"
            ],
            "medications": [
                "Lisinopril 10mg daily",
                "Metformin 500mg twice daily"
            ],
            "additional_context": "Patient is currently experiencing mild chest discomfort"
        },
        "active_question": None
    },
    "dave-ogle-patient-02": {
        "patient_context": {
            "patient_info": {
                "name": "davev ogle",
                "age": 38,
                "gender": "male"
            },
            "medical_history": [
                "Hypertension",
                "High cholesterol"
            ],
            "medications": [
                "Amlodipine 5mg daily",
                "Atorvastatin 20mg daily"
            ],
            "additional_context": "Patient is currently managing blood pressure"
        },
        "active_question": None
    },
    "izzy-patterson-patient-03": {
        "patient_context": {
            "patient_info": {
                "name": "izzy patterson",
                "age": 32,
                "gender": "female"
            },
            "medical_history": [
                "Paranoid Schizophrenia (diagnosed 8 years ago)",
                "Insomnia",
                "History of auditory hallucinations",
                "Previous psychiatric hospitalization (2 episodes)"
            ],
            "medications": [
                "Risperidone 4mg daily",
                "Clozapine 200mg at bedtime",
                "Lorazepam 1mg as needed for anxiety",
                "Trazodone 50mg for sleep"
            ],
            "additional_context": "Patient exhibits paranoid delusions about being monitored, occasional auditory hallucinations, and difficulty trusting medical staff. Recently reported feeling that her medication might be 'poisoned'. Shows social withdrawal and disorganized thinking during episodes. Relatively stable on current medication regimen but requires consistent follow-up."
        },
        "active_question": None
    }
}


def handle_question(question: str, channels: list[str]):
    result = {}
    
    for channel in channels:
        if channel not in active_channels:
            result[channel] = "channel doesn't exist"
        elif active_channels[channel]["active_question"] is not None:
            result[channel] = "currently a question being asked"
        else:
            # Get the patient's name from context
            patient_name = active_channels[channel]["patient_context"]["patient_info"]["name"]
            
            # Send the question to the patient's channel
            channel_obj = discord.utils.get(bot.get_all_channels(), name=channel)
            if channel_obj:
                asyncio.create_task(channel_obj.send(
                    f"Hello {patient_name},\n\n"
                    f"Your doctor has a question for you:\n"
                    f"**{question}**\n\n"
                    f"Please respond to this message with your answer."
                ))
            
            # start a conversation to ask the patient question
            patient_question = patientConversationAskQuestion(
                channel=channel,
                question=question,
                openAI_client=openai_client,
                discord_client=bot,
                context=active_channels[channel]["patient_context"]
            ) 
            
            # Mark the question as active for this channel
            active_channels[channel]["active_question"] = patient_question
            result[channel] = "Question sent"
    
    return {
        "status": result
    }


doctor_channel = doctor_conversation(
    channel=doctor_channel,
    openAI_client=openai_client,
    discord_client=bot,
    active_channels=active_channels,
    question_callback=handle_question
)

@bot.event
async def on_ready():
    """Send a startup message to the doctor's channel when the bot is ready"""
    print(f'Bot is ready! Logged in as {bot.user.name}')
    
    # Clean up all channels
    async def cleanup_channel(channel_name):
        channel = discord.utils.get(bot.get_all_channels(), name=channel_name)
        if channel:
            try:
                # Delete all messages in the channel
                async for message in channel.history(limit=None):
                    await message.delete()
                print(f"Cleaned up messages in channel: {channel_name}")
            except Exception as e:
                print(f"Error cleaning up channel {channel_name}: {str(e)}")
    
    # Clean up doctor's channel
    await cleanup_channel("dr-smith")
    
    # Clean up all patient channels
    for channel_name in active_channels.keys():
        await cleanup_channel(channel_name)
    
    # Find the doctor's channel and send welcome message
    doctor_channel_obj = discord.utils.get(bot.get_all_channels(), name="dr-smith")
    if doctor_channel_obj:
        await doctor_channel_obj.send(
            "Hello! I'm your healthcare assistant bot. I'm here to help you manage patient communications. "
            "I can help you:\n"
            "1. Ask questions to your patients\n"
            "2. Track patient responses\n"
            "3. Manage patient information\n\n"
            "How can I assist you today?"
        )
    else:
        print(f"Warning: Could not find doctor's channel 'dr-smith'")

@bot.event
async def on_message(message):
    """Handle incoming messages"""
    if message.author == bot.user:
        return

    channel_name = message.channel.name
    text = message.content

    if channel_name == "dr-smith":  # Check channel name instead of ID
        await doctor_channel.respond_to_message(text)
    
    elif channel_name in active_channels:
        if not (active_channels[channel_name]['active_question'] == None):
            patient_obj: patientConversationAskQuestion = active_channels[channel_name]['active_question']              
            await patient_obj.respond_to_message(text)

            if patient_obj.check_if_answered():
                await doctor_channel.update_question_answer(channel_name, patient_obj.question_answer, patient_obj.question)
                del active_channels[channel_name]['active_question']
                active_channels[channel_name]['active_question'] = None

    await bot.process_commands(message)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    bot.run(os.getenv('DISCORD_TOKEN'))  # Replace with your actual bot token