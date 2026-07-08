from dotenv import load_dotenv
load_dotenv()
import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
model = "gemma3:270m"


@traceable(run_type="tool")
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

@traceable(run_type="tool")
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


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model="gemma3:270m", tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    """
    Run the agent to answer a question using the provided tools.
    """
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    
    print(f"Agent received question: {question}")
    print("================================")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")
        response = ollama_chat_traced(messages=messages)
        
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("No tool calls detected. Final response:")
            return ai_message.content
        
        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool call detected: {tool_name} with arguments: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        observation = tool_to_use(**tool_args)

        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )
    print("Maximum iterations reached. Without a Final response:")    
    return None


if __name__ == "__main__":
    print("Starting the agent...")
    question = "What is the price of a laptop after applying a gold discount?"
    answer = run_agent(question)
    print(f"Final answer: {answer}")