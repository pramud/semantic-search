# pip install transformers torch accelerate sentencepiece bitsandbytes 


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json
import random
import time

# --- 1. Configuration & Model Loading ---
# <<< CHANGE: Updated MODEL_ID to Qwen2.5-7B-Instruct >>>
MODEL_ID = "Qwen/Qwen2-7B-Instruct" # Using the official Qwen2 Instruct model ID
# Optional: Quantization for lower memory usage (requires bitsandbytes)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True) # trust_remote_code often needed for Qwen
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    # quantization_config=bnb_config, # Uncomment if using quantization
    device_map="auto", # Automatically uses GPU if available
    torch_dtype=torch.bfloat16 # Recommended for Qwen2 for performance/compatibility
)
model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# Ensure pad token is set; if not, use eos_token (common practice)
if tokenizer.pad_token is None:
    print("Warning: pad_token not set, using eos_token as pad_token.")
    tokenizer.pad_token = tokenizer.eos_token

# Determine EOS token ID for generation stopping
generation_eos_token_id = tokenizer.eos_token_id


# --- 2. Define Dummy Functions ---
# (No changes needed in dummy functions)
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current weather for a given location."""
    print(f"--- Called Function: get_current_weather(location='{location}', unit='{unit}') ---")
    # Dummy implementation
    if not isinstance(location, str) or not location:
         return json.dumps({"error": "Location must be a non-empty string"})
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
    temp = random.randint(-10, 35) if unit == "celsius" else random.randint(14, 95)
    return json.dumps({
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": random.choice(conditions)
    })

def get_user_profile(user_id: str) -> str:
    """Gets profile information for a given user ID."""
    print(f"--- Called Function: get_user_profile(user_id='{user_id}') ---")
    # Dummy implementation
    if not isinstance(user_id, str) or not user_id:
         return json.dumps({"error": "User ID must be a non-empty string"})
    users = {
        "U123": {"name": "Alice", "city": "London", "interests": ["AI", "Hiking"]},
        "U456": {"name": "Bob", "city": "Paris", "interests": ["Cooking", "Music"]},
    }
    profile = users.get(user_id, {"error": f"User {user_id} not found"})
    return json.dumps(profile)

# Map function names to actual functions
AVAILABLE_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_user_profile": get_user_profile,
}

# --- 3. System Prompt Definition ---
# (Keeping the system prompt the same initially. Qwen2 Instruct should follow instructions well.
# If it fails to generate the desired format, this might need tweaking.)
SYSTEM_PROMPT = """You are a helpful assistant. You have access to the following tools to answer user questions.

Available Tools:
1.  **get_current_weather**:
    - Description: Gets the current weather for a given location.
    - Arguments:
        - location (string, required): The city or area (e.g., "London", "Paris").
        - unit (string, optional, default: "celsius"): Temperature unit ("celsius" or "fahrenheit").

2.  **get_user_profile**:
    - Description: Gets profile information for a given user ID.
    - Arguments:
        - user_id (string, required): The ID of the user (e.g., "U123").

To use a tool, you MUST respond ONLY with the following JSON format inside <tool_call> tags:
<tool_call>
{
  "name": "TOOL_NAME",
  "arguments": {
    "ARG_NAME_1": "VALUE_1",
    "ARG_NAME_2": "VALUE_2"
  }
}
</tool_call>

Do NOT add any other text, explanation, or conversational filler before or after the <tool_call> block if you decide to use a tool. Your response must start directly with `<tool_call>` if using a tool.
If you can answer the question directly without needing a tool, provide just the answer.

---

Examples:

User: What's the weather like in Hyderabad?
Assistant:
<tool_call>
{
  "name": "get_current_weather",
  "arguments": {
    "location": "Hyderabad"
  }
}
</tool_call>

User: tell me about user U456?
Assistant:
<tool_call>
{
  "name": "get_user_profile",
  "arguments": {
    "user_id": "U456"
  }
}
</tool_call>

User: What is the capital of France?
Assistant: The capital of France is Paris.
"""

# --- 4. Helper Function for LLM Interaction ---
def generate_response(prompt, max_new_tokens=150):
    """Generates text using the loaded model, attempting to use chat template."""
    # Use the tokenizer's chat template for Qwen's preferred format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        # Use apply_chat_template for Qwen's preferred format
        # add_generation_prompt=True adds the prompt structure for the Assistant's turn
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # Fallback to basic concatenation if template fails (less ideal)
        print("Falling back to basic prompt concatenation.")
        text = SYSTEM_PROMPT + "\n\nUser: " + prompt + "\nAssistant:"

    print(f"\nFormatted Prompt using Chat Template:\n{'-'*20}\n{text}\n{'-'*20}")

    # Tokenize the formatted prompt
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- Generating Initial Response ---")
    with torch.no_grad():
        # Use generate method with model_inputs (which are input_ids and attention_mask)
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, # Use the set pad_token_id
            eos_token_id=generation_eos_token_id, # Use the determined EOS token ID
            do_sample=True,
            temperature=0.6, # Qwen2 Instruct might be fine with slightly lower temp
            top_p=0.9
        )

    # Decode only the newly generated tokens
    # Handle potential token mismatch issues if generation starts unexpectedly
    start_index = model_inputs.input_ids.shape[1]
    response_ids = outputs[0][start_index:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(f"Model Raw Output:\n{'-'*20}\n{response_text}\n{'-'*20}")
    return response_text


def generate_final_response(prompt, tool_call_request, tool_result, max_new_tokens=200):
    """Generates the final response after a tool has been called, using chat template."""
    # Construct the message history including the tool call and result.
    # Using the 'tool' role which is standard for many models including recent Qwen versions.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        # Assistant's turn: the tool call it generated
        {"role": "assistant", "content": tool_call_request},
        # Tool's turn: the result from executing the function
        # <<< NOTE: Ensure tool_result is a JSON string if it came from our dummy functions >>>
        {"role": "tool", "content": tool_result},
    ]

    try:
        # Apply the chat template, prompting for the next assistant response
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
         print(f"Error applying chat template for final response: {e}")
         # Fallback (less ideal structure for the model)
         print("Falling back to basic concatenation for final response.")
         text = (
            SYSTEM_PROMPT +
            "\n\nUser: " + prompt +
            "\nAssistant: " + tool_call_request +
            "\nTool Result: " + tool_result + # Provide the result from the tool
            "\nAssistant:" # Ask the model to generate the final answer based on the result
         )

    print(f"\nFormatted Prompt for Final Response:\n{'-'*20}\n{text}\n{'-'*20}")

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- Generating Final Response ---")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode only the newly generated tokens
    start_index = model_inputs.input_ids.shape[1]
    response_ids = outputs[0][start_index:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(f"Model Final Raw Output:\n{'-'*20}\n{response_text}\n{'-'*20}")
    return response_text


# --- 5. Agent Logic ---
# (Parser should work, but watch Qwen2.5's output format closely)
def parse_tool_call(text: str):
    """Attempts to parse a tool call from the model's output."""
    # Regex to find the <tool_call> block and extract JSON content
    # Allows for potential whitespace variations and multi-line JSON
    match = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        try:
            # Basic unescaping (might need more robust handling if complex escapes occur)
            json_content_unescaped = json_content.replace('\\"', '"').replace("\\n", "\n").replace("\\'", "'").replace('\\\\', '\\')

            tool_info = json.loads(json_content_unescaped)
            # Basic validation
            if "name" in tool_info and "arguments" in tool_info and isinstance(tool_info["arguments"], dict):
                print(f"--- Parsed Tool Call: {tool_info} ---")
                return tool_info
            else:
                 print(f"--- Invalid JSON structure in tool call (missing keys or wrong types) ---")
                 print(f"--- JSON Content (Unescaped): {json_content_unescaped} ---")
                 return None
        except json.JSONDecodeError as e:
            print(f"--- JSON Parsing Error: {e} ---")
            print(f"--- Faulty JSON String: {json_content} ---")
            print(f"--- Faulty JSON String (Attempted Unescape): {json_content_unescaped} ---")

            # Attempt to fix common JSON issues like trailing commas (requires more robust parser or regex)
            # For now, just return None
            return None

    # Fallback check for direct JSON output (if model ignores tags)
    elif text.strip().startswith("{") and text.strip().endswith("}"):
         potential_json = text.strip()
         print("--- Detected potential JSON without <tool_call> tags. Attempting to parse... ---")
         try:
             # Basic unescaping
             potential_json_unescaped = potential_json.replace('\\"', '"').replace("\\n", "\n").replace("\\'", "'").replace('\\\\', '\\')
             tool_info = json.loads(potential_json_unescaped)
             # Check structure
             if "name" in tool_info and "arguments" in tool_info and isinstance(tool_info["arguments"], dict) and tool_info["name"] in AVAILABLE_FUNCTIONS:
                 print(f"--- Parsed Direct JSON Tool Call: {tool_info} ---")
                 return tool_info
             else:
                 print("--- Direct JSON does not match expected tool call structure. ---")
                 return None
         except json.JSONDecodeError as e:
             print(f"--- Direct JSON parsing failed: {e} ---")
             print(f"--- Direct JSON String: {potential_json} ---")
             print(f"--- Direct JSON String (Attempted Unescape): {potential_json_unescaped} ---")
             return None
    return None

def run_agent(user_query: str):
    """Main agent execution loop."""
    print(f"\n{'='*10} Processing Query: {user_query} {'='*10}")

    # 1. Initial Generation: Ask the model to respond or request a tool call
    initial_response = generate_response(user_query)

    # 2. Check for Tool Call
    tool_call_info = parse_tool_call(initial_response)

    final_answer = None # Initialize final_answer

    if tool_call_info:
        function_name = tool_call_info.get("name")
        arguments = tool_call_info.get("arguments", {})

        if function_name in AVAILABLE_FUNCTIONS:
            # 3. Execute the Function
            function_to_call = AVAILABLE_FUNCTIONS[function_name]
            try:
                tool_result = function_to_call(**arguments) # Assumes result is JSON string or simple type
                print(f"--- Tool Result: {tool_result} ---")

                # 4. Generate Final Response using the tool result
                # Pass the raw initial_response which contains the <tool_call> block
                final_answer = generate_final_response(user_query, initial_response, tool_result)

            except TypeError as e:
                print(f"--- Function Argument Error: {e} ---")
                error_message = f"Error calling function {function_name}: Invalid arguments provided. Details: {e}"
                # Pass the raw initial_response which contains the <tool_call> block
                final_answer = generate_final_response(user_query, initial_response, json.dumps({"error": error_message}))

            except Exception as e:
                print(f"--- Function Execution Error: {e} ---")
                error_message = f"Error executing function {function_name}: {e}"
                # Pass the raw initial_response which contains the <tool_call> block
                final_answer = generate_final_response(user_query, initial_response, json.dumps({"error": error_message}))

        else:
            print(f"--- Error: Model requested unknown function '{function_name}' ---")
            error_message = f"Error: You requested a tool named '{function_name}' which is not available."
            # Pass the raw initial_response which contains the <tool_call> block
            final_answer = generate_final_response(user_query, initial_response, json.dumps({"error": error_message}))

        # Return the generated final answer (might be based on tool result or error)
        return final_answer

    else:
        # 5. No Tool Call: Return the initial response directly
        print("--- No tool call detected. Returning direct response. ---")
        # The initial_response is assumed to be the final answer in this case.
        return initial_response

# --- 6. Example Usage ---
if __name__ == "__main__":
    queries = [
        "What's the weather like in London?",
        "Can you tell me about user U123?",
        "What is the capital of France?", # Should not trigger a tool call
        "Get the weather in Fahrenheit for New York",
        "Who is user U789?", # Should trigger tool call, result in error
        "Tell me the weather in Tokyo in celsius.", # Tool call with specific unit
    ]

    # Run only the first query for a quick test
    print("\n--- Running First Example Query ---")
    query = queries[0]
    start_time = time.time()
    agent_response = run_agent(query)
    end_time = time.time()
    print(f"\n>>> Final Agent Response for '{query}':\n{agent_response}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("=" * 60)


    # Interactive loop
    print("\nEnter your query (or type 'quit' to exit):")
    while True:
         try:
            user_input = input("> ")
         except EOFError: # Handle Ctrl+D or unexpected end of input
             print("\nExiting...")
             break
         if user_input.lower() == 'quit':
              break
         if not user_input:
             continue

         start_time = time.time()
         agent_response = run_agent(user_input)
         end_time = time.time()
         print(f"\n>>> Final Agent Response:\n{agent_response}")
         print(f"Time taken: {end_time - start_time:.2f} seconds")
         print("-" * 50)
