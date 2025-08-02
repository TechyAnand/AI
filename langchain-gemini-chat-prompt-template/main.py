from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create a message-based prompt
messages = [
    SystemMessage(
        content="You are KarTech Anand, a polyglot expert in Telugu, Hindi, English, and Spanish. "
                "You write organized content in a professional and assertive tone."
    ),
    HumanMessage(
        content="Please translate the sentence 'Good Morning. How are you?' to Telugu, Hindi, and Spanish languages"
    ),
]

# Invoke the model
result = model.invoke(messages)

# Print the result
print(f"\n🧠 Answer from AI:\n{result.content}")
