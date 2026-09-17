from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Explain concepts clearly and simply."
    ),
    (
        "human",
        "{question}"
    )
])

parser = StrOutputParser()

chain = prompt | model | parser

store = {}

def get_session_history(session_id):

    if session_id not in  store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]    

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_message_key = "question",
    history_message_key = "history"
)


print("=============================")
print("      LangChain AI Chatbot")
print("=============================")
print("Type 'exit' to stop\n")

session_id = "user_1"

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("GoodBye")
        break

    response = conversation_chain.invoke({
        "question": user_input
    },
    config = {
        "configurable":{
            "session_id": session_id
        }
    })

    print("AI:", response)