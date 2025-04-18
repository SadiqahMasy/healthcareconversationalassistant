import logging
from slack_bolt import App
from slack_sdk.web import WebClient
from openai import OpenAI

from doctor_conversation import doctor_conversation
from patient_conversation import patientConversationAskQuestion

# Initialize a Bolt for Python app
app = App()
slack_client = WebClient(token="your-slack-token")
openai_client = OpenAI()

# For simplicity we'll store our app data in-memory with the following data structure.
# onboarding_tutorials_sent = {"channel": {"user_id": OnboardingTutorial}}

doctor_channel = "doctor"

active_channels = {
    "patient-channel-1": {
        "patient_context": {
            "patient_info": {
                "name": "John Doe",
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
    "patient-channel-2": {
        "patient_context": {
            "patient_info": {
                "name": "Jane Smith",
                "age": 38,
                "gender": "female"
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
            
            # start a conversation to ask the patient question
            patient_question = patientConversationAskQuestion(
                channel=channel,
                question=question,
                openAI_client=openai_client,
                slack_client=slack_client,
                context= active_channels[channel]["patient_context"]
            ) 
            
            # Mark the question as active for this channel
            active_channels[channel]["active_question"] = patient_question
            result[channel] = "Question sent"
    
    return {
        "status": result
    }


doctor_channel = doctor_conversation(
    channel="doctor-channel",
    openAI_client=openai_client,
    slack_client=slack_client,
    active_channels=["patient-channel-1", "patient-channel-2"],
    question_callback=handle_question
)


# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@app.event("message")
def message(event, client):
    """Display the onboarding welcome message after receiving a message
    that contains "start".
    """
    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")

    if channel_id == doctor_channel:
        doctor_channel.respond_to_message(text)
    
    elif channel_id in active_channels:
        if not (active_channels[channel_id]['active_question'] == None):

            patient_obj: patientConversationAskQuestion = active_channels[channel_id]['active_question']              
            patient_obj.respond_to_message(text)

            if patient_obj.check_if_answered():
                doctor_channel.update_question_answer(channel_id, patient_obj.question_answer)
                active_channels[channel_id]['active_question'] = None                


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.start(3000)