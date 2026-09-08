import ollama


class ResponseGenerator:
    def __init__(
        self,
        model_name="llama3.2:1b"
    ):
        self.model_name = model_name


    def generate(
        self,
        emotion,
        confidence,
        user_text,
        conversation_history=None
    ):

        if conversation_history is None:
            conversation_history = []

        history_text = ""

        for turn in conversation_history[-2:]:
            history_text += (
                f"User: {turn['user']}\n"
                f"Robot: {turn['robot']}\n"
            )

        prompt = f"""
You are a small social robot having a conversation with a person.

Respond naturally and briefly.

The emotion classifier estimates the current user's emotion as:
Emotion: {emotion}
Confidence: {confidence:.2f}

Conversation history:
{history_text}

Current user utterance:
{user_text}

Instructions:
- Respond in one or two short sentences.
- Take the detected emotion into account.
- Do not mention the emotion classifier.
- Do not mention confidence scores.
- Do not explicitly say "you sound sad" unless it is natural.
- Be conversational and supportive.
- Avoid overly long explanations.
- If response with question make open ended questions to encourage further conversation.

Robot response:
"""

        response = ollama.chat(
            model = self.model_name, 
            messages = [
                {
                    'role': 'user', 
                    'content': prompt
                }])


        return response.message.content.strip()