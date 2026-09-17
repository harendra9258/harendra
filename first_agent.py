from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid mathematical expression."


@tool
def get_weather(city: str) -> str:
    """Get the weather of a city."""

    weather_data = {
        "hyderabad": "32°C, Sunny",
        "delhi": "30°C, Cloudy",
        "mumbai": "28°C, Rainy",
        "bangalore": "25°C, Pleasant"
    }

    return weather_data.get(
        city.lower(),
        "Weather information not available."
    )


tools = [calculator, get_weather]

tool_dict = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)

parser = StrOutputParser()


print("==============================")
print("        Tool Chatbot")
print("==============================")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages = [HumanMessage(content=user_input)]

    response = model_with_tools.invoke(messages)

    if response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nCalling tool: {tool_name}")
            print(f"Arguments: {tool_args}")

            selected_tool = tool_dict[tool_name]

            tool_result = selected_tool.invoke(tool_args)

            print(f"Tool result: {tool_result}")

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

        final_response = model_with_tools.invoke(messages)

        final_answer = parser.invoke(final_response)

        print("\nAI:", final_answer)

    else:

        print("\nAI:", parser.invoke(response))