from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
model = "gemma3:270m"


@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product by its name.
    """
    print(f"Looking up price for product: {product_name}")
    # Simulate a product price lookup
    product_prices = {
        "laptop": "999",
        "smartphone": "699",
        "headphones": "199",
        "monitor": "299",
        "keyboard": "49"
    }
    return product_prices.get(product_name.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a given price based on the discount tier.
    Available tiers are silver,bronze,gold
    """
    print(f"Applying {discount_tier} discount to price: {price}")
    discount_rates = {
        "silver": 12,
        "gold": 23,
        "bronze": 5
    }
    discount = discount_rates.get(discount_tier.lower(), 0.0)
    discounted_price = round(price * (1 - discount / 100), 2)
    return discounted_price

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    """
    Run the agent to answer a question using the provided tools.
    """
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    # llm = init_chat_model(model="ollama:gemma3:270m", temperature=0)
    llm = init_chat_model(
                        model="gemini-2.5-flash",
                        model_provider="google_genai",
                        temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Agent received question: {question}")
    print("================================")

    messages = [

        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "you have access to product catalog tool"
                "and a discount tool"
                "Strict Rules"
                "1.Never guess the price of a product, always use the product catalog tool to get the price"
                "you must call the get_product_price tool to get the price of a product"
                "2.only call apply_discount tool if you have the price of the product from the get_product_price tool"
                "do not pass a made up number to the apply_discount tool, always use the price from the get_product_price tool"
                "3.never calculate the price of a product manually, always use the apply_discount tool to calculate the discounted price"
                "4. if the user did not mention a discount tier, ask the user to specify a discount tier before applying the discount"
            )
            
        )   ,
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")
        ai_response = llm_with_tools.invoke(messages)
        
        tool_calls = ai_response.tool_calls

        if not tool_calls:
            print("No tool calls detected. Final response:")
            return ai_response.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call detected: {tool_name} with arguments: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_response)
        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        )
        messages.append(tool_message)

    print("Maximum iterations reached. Without a Final response:")    
    return None


if __name__ == "__main__":
    print("Starting the agent...")
    question = "What is the price of a laptop after applying a gold discount?"
    answer = run_agent(question)
    print(f"Final answer: {answer}")