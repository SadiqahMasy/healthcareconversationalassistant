import discord
from discord.ext import commands
import asyncio
import io
import wave
import numpy as np
import torch
import torchaudio
import whisper
import tempfile
import os
from pydub import AudioSegment
import threading
import queue
import webrtcvad
import struct
import datetime
from dotenv import load_dotenv

from csm.generator import Segment, load_csm_1b
from csm.utils import prepare_prompt
from llm_adapter import OllamaAdapter

# Load environment variables from .env file
load_dotenv()

# Setup csm
os.environ["NO_TORCH_COMPILE"] = "1"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"[Torch] Using device: {device}")


# Bot configuration
intents = discord.Intents.default()
intents.message_content = True 
intents.voice_states = True
bot = commands.Bot(command_prefix='!', intents=intents)


# Initialize Whisper model
whisper_model = whisper.load_model("base")

# CSM
generator = load_csm_1b(device)

# LLM
MODEL = "llama3.2:3b-instruct-q4_0"
llm_adapter = OllamaAdapter()

def generate_chat(messages: list[dict]):
    return llm_adapter.chat(
        messages,
        model=MODEL
    )['message']['content']

# Initialize VAD
vad = webrtcvad.Vad(3)  # Aggressiveness from 0 to 3 (3 is most aggressive)

# Queue for audio processing
audio_queue = queue.Queue()

# Buffer for storing audio frames
speaking_buffer = []
silence_frames = 0
is_speaking = False

# Tracking variables for recording sessions
recording_sessions = {}  # voice_client -> should_stop flag

# Output file for transcriptions
TRANSCRIPTION_FILE = "transcriptions.txt"
AUDIO_OUTPUT_DIR = "recorded_audio"


# Global conversation data
conversation_archive: list [Segment] = []


# Ensure audio directory exists
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

def log_transcription(username, text, audio_file: bytes=None):
    """Log transcription to file with timestamp and audio filename if available"""
    
    print(f"[Log] Saved transcription from {username} to {text}")

    return [prepare_prompt(
        text,
        username,
        audio_file,
        generator.sample_rate
    )]
    

@bot.event
async def on_ready():
    print(f'[Bot] Bot is ready. Logged in as {bot.user}')
    print(f'[Bot] Starting audio processing thread...')
    threading.Thread(target=process_audio_queue, daemon=True).start()
    print(f'[Bot] Audio processing thread started')

@bot.command()
async def join(ctx):
    """Join the user's voice channel"""
    print(f'[Command] !join command received from {ctx.author.name}')
    if ctx.author.voice:
        print(f'[Voice] User {ctx.author.name} is in voice channel {ctx.author.voice.channel.name}')
        channel = ctx.author.voice.channel
        try:
            print(f'[Voice] Attempting to connect to {channel.name}')
            voice_client = await channel.connect()
            print(f'[Voice] Successfully connected to {channel.name}')
            await ctx.send(f"Joined {channel.name}")
            
            # Start real-time audio processing with VAD
            print(f'[Voice] Starting VAD listening in {channel.name}')
            await start_vad_listening(ctx, voice_client)
        except Exception as e:
            print(f'[Voice] Error connecting to voice channel: {e}')
            await ctx.send("Failed to join voice channel. Please try again.")
    else:
        print(f'[Voice] User {ctx.author.name} is not in a voice channel')
        await ctx.send("You need to be in a voice channel first!")

async def start_vad_listening(ctx, voice_client):
    """Set up real-time audio processing with VAD"""
    print(f'[Voice] Setting up VAD listening in {voice_client.channel.name}')
    try:
        # Register this voice client in the recording sessions
        recording_sessions[voice_client] = False
        
        # Configure the sink to record audio
        sink = discord.sinks.WaveSink()
        print(f'[Voice] Created WaveSink for recording')
        
        async def on_recording_finished(recorded_sink, *args):
            try:
                print(f'[Voice] Recording finished callback triggered with {len(args)} additional args')
                # Process all accumulated audio with VAD
                await process_with_vad(ctx, recorded_sink)
            except Exception as e:
                print(f"[Voice] Error in recording finished callback: {e}")
        
        # Process audio in smaller chunks for VAD analysis
        frame_duration = 10  # Process 10 seconds at a time
        print(f'[Voice] Set frame duration to {frame_duration}s')
        
        global speaking_buffer, silence_frames, is_speaking
        speaking_buffer = []
        silence_frames = 0
        is_speaking = False
        print(f'[Voice] Initialized speaking buffer and state')
        
        # Use a simpler approach: record in fixed chunks of time
        max_iterations = 1440  # 4 hours (1440 * 10s = 14400s = 4h)
        
        print(f'[Voice] Starting recording loop (max {max_iterations} iterations)')
        
        for iteration in range(max_iterations):
            # Check if we should stop (signal set from another thread/function)
            if recording_sessions.get(voice_client, True):
                print("[Voice] Received stop signal, ending recording loop")
                break
            
            # Check if still connected
            if not voice_client.is_connected():
                print("[Voice] Voice client disconnected, stopping VAD listening")
                break
            
            try:
                print(f'[Voice] Starting recording iteration {iteration}')
                # Start a new recording
                voice_client.start_recording(
                    sink,
                    on_recording_finished,
                    ctx
                )
                
                # Wait for the frame duration, checking for stop signal
                for _ in range(int(frame_duration * 2)):  # Check every 0.5 seconds
                    if recording_sessions.get(voice_client, True) or not voice_client.is_connected():
                        print("[Voice] Detected stop signal during sleep")
                        break
                    await asyncio.sleep(0.5)
                
                # Check if still connected
                if not voice_client.is_connected() or recording_sessions.get(voice_client, True):
                    print("[Voice] Voice client disconnected or stop signal received during recording")
                    break
                
                print(f'[Voice] Stopping recording iteration {iteration}')
                
                # Check if we're still recording before trying to stop
                if voice_client.recording:
                    voice_client.stop_recording()
                
                # Create a new sink for the next iteration
                sink = discord.sinks.WaveSink()
                
                # Brief pause between recordings
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"[Voice] Error in listening loop: {e}")
                
                # Try to recover by creating a new sink and continuing
                try:
                    if voice_client.recording:
                        voice_client.stop_recording()
                except Exception as stop_error:
                    print(f"[Voice] Failed to stop recording: {stop_error}")
                
                sink = discord.sinks.WaveSink()
                await asyncio.sleep(1)  # Wait a bit before continuing
                continue
                
        print("[Voice] Exited listening loop")
        # Final cleanup
        try:
            if voice_client.is_connected() and voice_client.recording:
                voice_client.stop_recording()
        except Exception as e:
            print(f"[Voice] Error during final cleanup: {e}")
        
        # Remove from tracking
        if voice_client in recording_sessions:
            del recording_sessions[voice_client]
            
    except Exception as e:
        print(f"[Voice] Error in start_vad_listening: {e}")
        try:
            if voice_client.is_connected() and voice_client.recording:
                voice_client.stop_recording()
        except Exception as nested_e:
            print(f"[Voice] Error stopping recording during exception handling: {nested_e}")
        
        # Remove from tracking on error
        if voice_client in recording_sessions:
            del recording_sessions[voice_client]
            
        if voice_client.is_connected():
            await voice_client.disconnect()

async def process_with_vad(ctx, sink):
    """Process audio with VAD to determine if someone is speaking"""
    global speaking_buffer, silence_frames, is_speaking
    
    try:
        # Check if we have any audio data
        if not sink or not hasattr(sink, 'audio_data') or not sink.audio_data:
            print("[Voice] No valid audio data in sink")
            return
            
        print(f"[Voice] Processing audio from {len(sink.audio_data)} users")
        
        if ctx.voice_client.is_playing():
            ctx.voice_client.stop()
            
        # Process each user's audio
        for user_id, audio_data in sink.audio_data.items():
            # Skip the bot's own audio
            if user_id == bot.user.id:
                continue
                
            try:
                # Make sure the audio data has a file attribute
                if not hasattr(audio_data, 'file'):
                    print(f"[Voice] Audio data for user {user_id} has no file attribute")
                    continue
                    
                # Get raw audio data
                try:
                    audio_data.file.seek(0)
                    raw_data = audio_data.file.read()
                except Exception as e:
                    print(f"[Voice] Error reading audio file: {e}")
                    continue
                
                if not raw_data or len(raw_data) < 1000:  # Skip very small chunks
                    print(f"[Voice] Skipping small audio chunk ({len(raw_data) if raw_data else 0} bytes)")
                    continue
                
                # Get user information
                try:
                    user = await bot.fetch_user(user_id)
                    username = user.name if user else f"User {user_id}"
                except Exception as e:
                    print(f"[Voice] Error fetching user {user_id}: {e}")
                    username = f"User {user_id}"
                
                print(f"[Voice] Processing {len(raw_data)} bytes of audio from {username}")
                
                # Create a temporary file for Whisper processing
                try:
                    # Instead of using VAD for small chunks, process longer segments directly
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        # Create a proper WAV file with header
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            # Set the parameters for the WAV file
                            wav_file.setnchannels(2)  # Stereo
                            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                            wav_file.setframerate(48000)  # 48kHz sample rate
                            wav_file.writeframes(raw_data)
                            
                        temp_filename = temp_file.name
                    
                    # Generate a unique filename for saving the audio
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = os.path.join(AUDIO_OUTPUT_DIR, f"{timestamp}_{username.replace(' ', '_')}.wav")
                    
                    # Save a copy of the audio file
                    try:
                        with open(temp_filename, 'rb') as src_file:
                            with open(output_filename, 'wb') as dest_file:
                                dest_file.write(src_file.read())
                        print(f"[Voice] Saved audio to {output_filename}")
                    except Exception as e:
                        print(f"[Voice] Error saving audio file: {e}")
                    
                    # Add to queue for processing with Whisper
                    audio_queue.put((temp_filename, ctx, username))
                    await ctx.send(f"Processing speech from {username}...")
                except Exception as e:
                    print(f"[Voice] Error creating temporary file: {e}")
                    continue
                
            except Exception as e:
                print(f"[Voice] Error processing audio for user {user_id}: {e}")
                continue
                
    except Exception as e:
        print(f"[Voice] Error in process_with_vad: {e}")
        # Reset state on error
        speaking_buffer = []
        silence_frames = 0
        is_speaking = False

def process_audio_queue():
    global conversation_archive
    """Process audio files in the queue with Whisper"""
    while True:
        try:
            # Get the next audio file from the queue
            temp_filename, ctx, username = audio_queue.get()
            print(f"[Whisper] Starting transcription for {username} from file {temp_filename}")
            
            # Generate a unique filename for saving the audio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(AUDIO_OUTPUT_DIR, f"{timestamp}_{username.replace(' ', '_')}.wav")
            

            # Read the audio file into a bytes buffer
            try:
                with open(temp_filename, 'rb') as src_file:
                    audio_bytes = src_file.read()
            except Exception as e:
                print(f"[Whisper] Error saving audio file: {e}")
                audio_bytes = None

            
            # Process with Whisper
            print("[Whisper] Running model...")
            result = whisper_model.transcribe(temp_filename, fp16=False)
            print(f"[Whisper] Model completed. Raw result: {result}")
            
            transcribed_text = result["text"].strip()
            print(f"[Whisper] Extracted text: '{transcribed_text}'")
            
            if transcribed_text:
                # Log the transcription to file with audio file reference
                if audio_bytes != None:
                    conversation_archive += log_transcription(0, transcribed_text, audio_bytes)
                else:
                    print("[Whisper] failed to get audio file bytes")
                
                # Send the transcription to the Discord channel
                # print(f"[Whisper] Sending transcription to Discord for {username}")
                # asyncio.run_coroutine_threadsafe(
                #     ctx.send(f"{username} said: {transcribed_text}"), bot.loop
                # )
                response = generate_chat(
                    [{"role": ("assistant" if c.speaker == 1 else "user"), "content": c.text} for c in conversation_archive]
                )
                print(f"[Ollama] {response}")

                audio_tensor = generator.generate(
                    text=response,
                    speaker=0,
                    context=conversation_archive,
                    max_audio_length_ms=10_000
                )

                # Convert to byte stream instead of saving to file
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    audio_tensor.cpu(),
                    generator.sample_rate,
                    format="wav"
                )
                buffer.seek(0)
                audio_bytes = buffer.read()
                print(f"[CSM] Successfully generated audio response ({len(audio_bytes)} bytes)")
                
                # Play the audio in the voice channel instead of sending as a file
                if ctx.voice_client and ctx.voice_client.is_connected():
                    # Save to a temporary file since FFmpegPCMAudio needs a file path
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_file.write(audio_bytes)
                        temp_filename = temp_file.name
                    
                    # Make sure the bot isn't already playing audio
                    if ctx.voice_client.is_playing():
                        ctx.voice_client.stop()
                    
                    # Play the audio
                    audio_source = discord.FFmpegPCMAudio(temp_filename)
                    ctx.voice_client.play(audio_source, after=lambda e: os.remove(temp_filename) if e is None else print(f"[Voice] Error playing audio: {e}"))
                    print(f"[Voice] Playing audio response in {ctx.voice_client.channel.name}")
                else:
                    # If not in a voice channel, send as a file instead
                    audio_file = discord.File(io.BytesIO(audio_bytes), filename="response.wav")
                    asyncio.run_coroutine_threadsafe(
                        ctx.send("I'm not in a voice channel, here's the audio file:", file=audio_file), bot.loop
                    )
                
                # save models
                conversation_archive += log_transcription(1, transcribed_text, audio_bytes)

            else:
                print("[Whisper] No text was transcribed")
                
            # Clean up the temporary file
            print(f"[Whisper] Cleaning up temporary file {temp_filename}")
            os.remove(temp_filename)
            audio_queue.task_done()
            
        except Exception as e:
            print(f"[Whisper] Error processing audio: {e}")
            print(f"[Whisper] Error details: {type(e).__name__}: {str(e)}")

@bot.command()
async def leave(ctx):
    """Leave the voice channel"""
    print(f'[Command] !leave command received from {ctx.author.name}')
    if ctx.voice_client:
        print(f'[Voice] Disconnecting from {ctx.voice_client.channel.name}')
        
        # Signal the recording loop to stop
        if ctx.voice_client in recording_sessions:
            recording_sessions[ctx.voice_client] = True
            print(f'[Voice] Set stop signal for recording session')
            
            # Give the recording loop a moment to clean up
            await asyncio.sleep(1)
        
        # Stop recording if it's active
        try:
            if ctx.voice_client.recording:
                ctx.voice_client.stop_recording()
                print(f'[Voice] Stopped recording in {ctx.voice_client.channel.name}')
        except Exception as e:
            print(f'[Voice] Error stopping recording: {e}')
        
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel")
    else:
        print(f'[Voice] Not in a voice channel')
        await ctx.send("I'm not in a voice channel")

@bot.event
async def on_voice_state_update(member, before, after):
    """Handle voice state updates"""
    if member == bot.user:
        if after.channel is None:  # Bot left voice channel
            print("[Voice] Bot disconnected from voice channel")
            # Clean up any ongoing recordings
            if hasattr(bot, 'voice_clients'):
                for vc in bot.voice_clients:
                    # Signal recording to stop if there's an active session
                    if vc in recording_sessions:
                        recording_sessions[vc] = True
                        print(f"[Voice] Set stop signal for voice client in {vc.channel}")
                    
                    if vc.is_connected():
                        try:
                            if vc.recording:
                                vc.stop_recording()
                                print("[Voice] Stopped recording due to disconnection")
                        except Exception as e:
                            print(f"[Voice] Error stopping recording: {e}")
                        
                        # Force disconnect
                        await vc.disconnect()

@bot.command()
async def show_transcripts(ctx, lines: int = 10):
    """Show the most recent transcriptions"""
    print(f'[Command] !show_transcripts command received from {ctx.author.name}')
    try:
        if not os.path.exists(TRANSCRIPTION_FILE):
            await ctx.send("No transcriptions have been saved yet.")
            return
            
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            transcript_lines = f.readlines()
            
        if not transcript_lines:
            await ctx.send("Transcription file exists but is empty.")
            return
            
        # Get the last N lines
        recent_lines = transcript_lines[-lines:]
        transcript_text = "".join(recent_lines)
        
        # If too long, send as a file
        if len(transcript_text) > 1900:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_file.write(transcript_text.encode('utf-8'))
                temp_filename = temp_file.name
                
            await ctx.send(f"Last {lines} transcriptions:", file=discord.File(temp_filename, "recent_transcripts.txt"))
            os.remove(temp_filename)
        else:
            await ctx.send(f"**Last {lines} transcriptions:**\n```\n{transcript_text}\n```")
    except Exception as e:
        print(f"[Command] Error showing transcripts: {e}")
        await ctx.send(f"Error showing transcripts: {str(e)}")

@bot.command()
async def list_recordings(ctx):
    """List all recorded audio files"""
    print(f'[Command] !list_recordings command received from {ctx.author.name}')
    try:
        if not os.path.exists(AUDIO_OUTPUT_DIR):
            await ctx.send("No recordings directory found.")
            return
            
        recordings = os.listdir(AUDIO_OUTPUT_DIR)
        recordings = [f for f in recordings if f.endswith('.wav')]
        
        if not recordings:
            await ctx.send("No audio recordings found.")
            return
            
        # Sort by most recent
        recordings.sort(reverse=True)
        
        # Limit to 20 files to avoid message size limits
        recordings = recordings[:20]
        
        files_list = "\n".join(recordings)
        await ctx.send(f"**Recent audio recordings:**\n```\n{files_list}\n```")
    except Exception as e:
        print(f"[Command] Error listing recordings: {e}")
        await ctx.send(f"Error listing recordings: {str(e)}")

@bot.command()
async def play_recording(ctx, filename):
    """Play back a recorded audio file"""
    print(f'[Command] !play_recording command received from {ctx.author.name} for file: {filename}')
    try:
        # Check if the file exists
        file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            await ctx.send(f"File not found: {filename}")
            return
            
        # Join the voice channel of the user
        if ctx.author.voice:
            # If already in a voice channel, move to the user's channel
            if ctx.voice_client:
                if ctx.voice_client.channel != ctx.author.voice.channel:
                    await ctx.voice_client.move_to(ctx.author.voice.channel)
            else:
                await ctx.author.voice.channel.connect()
                
            # Play the audio file
            source = discord.FFmpegPCMAudio(file_path)
            ctx.voice_client.play(source)
            
            await ctx.send(f"Playing recording: {filename}")
        else:
            await ctx.send("You need to be in a voice channel to use this command.")
    except Exception as e:
        print(f"[Command] Error playing recording: {e}")
        await ctx.send(f"Error playing recording: {str(e)}")

@bot.command()
async def search_transcripts(ctx, *, search_term):
    """Search for a term in the transcription file"""
    print(f'[Command] !search_transcripts command received from {ctx.author.name}: {search_term}')
    try:
        if not os.path.exists(TRANSCRIPTION_FILE):
            await ctx.send("No transcriptions have been saved yet.")
            return
            
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            transcript_lines = f.readlines()
            
        if not transcript_lines:
            await ctx.send("Transcription file exists but is empty.")
            return
            
        # Search for the term in each line
        matching_lines = [line for line in transcript_lines if search_term.lower() in line.lower()]
        
        if not matching_lines:
            await ctx.send(f"No matches found for '{search_term}'")
            return
            
        # Format the results
        result_text = f"Found {len(matching_lines)} matches for '{search_term}':\n\n"
        result_text += "".join(matching_lines)
        
        # If too long, send as a file
        if len(result_text) > 1900:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_file.write(result_text.encode('utf-8'))
                temp_filename = temp_file.name
                
            await ctx.send(f"Search results for '{search_term}':", 
                          file=discord.File(temp_filename, "search_results.txt"))
            os.remove(temp_filename)
        else:
            await ctx.send(f"**Search results:**\n```\n{result_text}\n```")
    except Exception as e:
        print(f"[Command] Error searching transcripts: {e}")
        await ctx.send(f"Error searching transcripts: {str(e)}")

# Run the bot with your token
bot.run(os.getenv('DISCORD_TOKEN'))