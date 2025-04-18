import openai
from typing import List, Callable, Dict
import discord
from discord.ext import commands

class doctor_conversation:

    def __init__(self, channel: str, openAI_client, discord_client: commands.Bot, active_channels: List[str], question_callback: Callable[[str, List[str]], Dict]) -> None:
        self.channel = channel  # Doctor's channel
        self.openAI_client = openAI_client
        self.discord_client = discord_client
        self.active_channels = active_channels
        self.question_callback = question_callback
        self.pending_questions: Dict[str, str] = {}  # channel -> question
        self.answered_questions: Dict[str, str] = {}  # channel -> answer
        self.functions = [
            {
                "name": "ask_patient_question",
                "description": "Ask a question to patients in specified channels",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask patients"
                        },
                        "channels": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of channels to ask the question in"
                        }
                    },
                    "required": ["question", "channels"]
                }
            }
        ]
        self.message_history = []

    async def respond_to_message(self, new_message: str):
        # Add new message to history
        self.message_history.append({"role": "user", "content": new_message})
        
        # Prepare messages for ChatGPT with context about active channels
        system_message = f"""You are a helpful healthcare assistant helping doctors manage patient outreach. 
        Available channels for questions: {', '.join(self.active_channels)}.
        When the doctor wants to ask patients questions, use the ask_patient_question function.
        Make sure to only use channels from the available list."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": new_message}
        ]
        messages.extend(self.message_history)
        
        response = self._call_chatgpt(messages)
        self.message_history.append({"role": "assistant", "content": response})
        
        return await self.message_doctor(response)

    def _call_chatgpt(self, messages):
        response = self.openAI_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=self.functions,
            function_call="auto"
        )
        
        response_message = response.choices[0].message
        
        if response_message.function_call:
            function_name = response_message.function_call.name
            if function_name == "ask_patient_question":
                import json
                function_args = json.loads(response_message.function_call.arguments)
                result = self.question_callback(function_args["question"], function_args["channels"])
                
                # Add function result to context
                self.message_history.append({
                    "role": "function",
                    "name": "ask_patient_question",
                    "content": json.dumps(result)
                })
                
                # Get a new response with the function result
                response = self._call_chatgpt(messages + [response_message, self.message_history[-1]])
                return response
        
        return response_message.content

    async def update_question_answer(self, channel: str, answer: str, question: str):
        """Called when a patient answers a question"""
    
        # Notify doctor of the answer
        summary = f"Question in {channel}: {question}\nAnswer: {answer}"
        await self.send_message(self.channel, summary)

    async def send_message(self, channel: str, msg: str):
        """Send a message to a specific channel using Discord"""
        try:
            # Find channel by name
            channel_obj = discord.utils.get(self.discord_client.get_all_channels(), name=channel)
            if channel_obj:
                await channel_obj.send(msg)
                return True
            print(f"Warning: Could not find channel '{channel}'")
            return False
        except Exception as e:
            print(f"Error sending message to {channel}: {str(e)}")
            return None

    async def message_doctor(self, msg: str):
        """Send a message to the doctor's channel"""
        return await self.send_message(self.channel, msg)

