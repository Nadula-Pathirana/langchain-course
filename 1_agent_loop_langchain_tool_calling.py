from dotenv import load_dotenv
from openai.types.responses import \
    response_computer_tool_call_output_screenshot

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# pyrefly: ignore [missing-import]
from langsmith import traceable

llm = init_chat_model("openrouter:apodex/apodex-1.1-mini:free", temperature=0)

MAX_ITERATIONS = 10

@tool
def get_product_price(product:str) -> float:
    """Look up the price of a product in the catalog"""
    print(f".  >> Executing get_product_price (product: '{product}')")
    prices ={"Laptop":1299.99,"headphones":149.95,"Keyboard":89.50}
    return prices.get(product,0.0)

@tool
def apply_discount(price:float ,discount_tier:str) ->float :
    """Apply a discount tier to a price and return the final price.
    Available tiers: Bronze, Silver, Gold."""
    print(f".  >> Executing apply_discount(price: '{price}', discount_tier: '{discount_tier}')")
    discount_percentages = {"Bronze":5,"Silver":12,"Gold":23}
    discount = discount_percentages.get(discount_tier,0.0)
    return round(price*(1-discount/100),2)

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(user_query:str):
    tools = [get_product_price,apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model("openrouter:apodex/apodex-1.1-mini:free", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question : {user_query}")
    print("=" * 70)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant." 
                "You have access to a product catalog tool."
                "And a discount tool\n\n"
                "Strict rules: you must follow this exactly.:\n"
                "1. Never guess or assume any product price."
                "You must call `get product price` first to get the real price. "
                "2. only call `apply discount` after you have received a price from `get product price`. Pass the exact price. "
                "Return by getting the product price. Do not pass a made-up number. "
                "3. Never calculate discount yourself using math."
                "Always use the apply discount tool."
                "4. If the user does not specify a discount tier, ask them which tier to use. Do not assume one."

            )
        ),
        HumanMessage(content=user_query),
    ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n ---Iteration {iteration}----\n")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"final answer : {ai_message.content}")
            return ai_message.content

        tool_call =tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args',{})
        tool_call_id = tool_call.get('id')
        
        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Error")
        observation = tool_to_use.invoke(tool_args)
        
        print(f" [Tool Result] {observation}")
            
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Error :Max iterations Reached without final answer")
    return None
    

        
    


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying gold discount?")
    print(result)
    

    


    

    

