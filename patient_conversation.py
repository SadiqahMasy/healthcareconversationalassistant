import openai
from slack_sdk.web import WebClient
from typing import Dict, Optional

class patientConversationAskQuestion:

    def __init__(self, channel: str, question: str, openAI_client, slack_client: WebClient, context: Optional[Dict] = None) -> None:
        self.channel = channel
        self.question = question
        self.openAI_client = openAI_client
        self.slack_client = slack_client
        self.context = context or {}
        self.question_answer = None
        self.message_history = []
        self.functions = [
            {
                "name": "question_answered",
                "description": "Call this function when the patient's question has been fully answered",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A brief summary of the answer provided"
                        }
                    },
                    "required": ["summary"]
                }
            }
        ]

    def respond_to_message(self, new_message: str):
        # Add new message to history
        self.message_history.append({"role": "user", "content": new_message})
        
        # Prepare system message with context
        system_message = self._build_system_message()
        
        # Prepare messages for ChatGPT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Original question: {self.question}"}
        ]
        messages.extend(self.message_history)
        
        response = self._call_chatgpt(messages)
        self.message_history.append({"role": "assistant", "content": response})
        
        return self.send_message(self.channel, response)

    def _build_system_message(self) -> str:
        """Build the system message incorporating context"""
        base_message = "You are a helpful healthcare assistant. Answer questions clearly and concisely. If you fully answer a question, call the question_answered function."
        
        if not self.context:
            return base_message
            
        context_parts = []
        
        # Add patient information if available
        if "patient_info" in self.context:
            patient_info = self.context["patient_info"]
            info_parts = []
            if "name" in patient_info:
                info_parts.append(f"Patient's name is {patient_info['name']}")
            if "age" in patient_info:
                info_parts.append(f"Patient is {patient_info['age']} years old")
            if "gender" in patient_info:
                info_parts.append(f"Patient's gender is {patient_info['gender']}")
            if info_parts:
                context_parts.append("Patient Information: " + ", ".join(info_parts))
        
        # Add medical history if available
        if "medical_history" in self.context:
            history = self.context["medical_history"]
            if isinstance(history, list):
                context_parts.append("Medical History: " + ", ".join(history))
            elif isinstance(history, str):
                context_parts.append("Medical History: " + history)
        
        # Add current medications if available
        if "medications" in self.context:
            meds = self.context["medications"]
            if isinstance(meds, list):
                context_parts.append("Current Medications: " + ", ".join(meds))
            elif isinstance(meds, str):
                context_parts.append("Current Medications: " + meds)
        
        # Add any additional context
        if "additional_context" in self.context:
            context_parts.append("Additional Context: " + str(self.context["additional_context"]))
        
        if context_parts:
            return f"{base_message}\n\nContext:\n" + "\n".join(context_parts)
        
        return base_message

    def _call_chatgpt(self, messages):
        response = self.openAI_client.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=self.functions,
            function_call="auto"
        )
        
        response_message = response.choices[0].message
        
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            if function_name == "question_answered":
                import json
                function_args = json.loads(response_message["function_call"]["arguments"])
                self.question_answered(function_args["summary"])
        
        return response_message["content"]

    def question_answered(self, question_answer: str):
        self.question_answer = question_answer

    def check_if_answered(self):
        return not (self.question_answer == None)

    def send_message(self, channel: str, msg: str):
        """Send a message to a specific channel using Slack WebClient"""
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=msg,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": msg
                        }
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Error sending message to {channel}: {str(e)}")
            return None

