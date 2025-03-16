from transformers import AutoTokenizer, Gemma3ForCausalLM #, BitsAndBytesConfig
import torch
import time

model_id = "google/gemma-3-1b-it"

# Load model and tokenizer only once globally
def initialize_model():
    """Initialize the Gemma model and tokenizer."""
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = Gemma3ForCausalLM.from_pretrained(
        model_id #, quantization_config=quantization_config
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

# Global model and tokenizer instances
model, tokenizer = initialize_model()

def get_code_summary(code_text, file_path=None, loading_callback=None):
    """
    Get a summary of code from Gemma.
    
    Args:
        code_text (str): The code to summarize
        file_path (str, optional): The file path for context
        loading_callback (callable, optional): A callback function to show loading progress
        
    Returns:
        str: The generated summary
    """
    # Limit text length if needed to avoid context window limitations
    # if len(code_text) > 8000:
    #     code_text = code_text[:8000] + "... [truncated]"
    
    # Create prompt
    context = f"from file {file_path}: " if file_path else ""
    prompt = f"Summarize this code snippet {context}concisely in 2-3 sentences. Focus on what it does, not how it does it:\n\n{code_text}"
    
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful code assistant that provides brief, accurate summaries of code."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt},]
            },
        ],
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device).to(torch.bfloat16)
    
    # Show loading animation if callback provided
    start_time = time.time()
    
    with torch.inference_mode():
        # Call loading callback periodically during generation
        if loading_callback:
            def loading_progress_callback(beam_idx, token_idx, token_id, **kwargs):
                if token_idx % 5 == 0:  # Update animation every 5 tokens
                    elapsed = time.time() - start_time
                    loading_callback(elapsed)
                return True
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,
                callback_function=loading_progress_callback
            )
        else:
            outputs = model.generate(**inputs, max_new_tokens=200)
    
    response = tokenizer.batch_decode(outputs)[0]
    
    # Clean up the response
    # Remove the input prompt and get just the model's response
    response_parts = response.split("[/INST]")
    if len(response_parts) > 1:
        summary = response_parts[1].strip()
    else:
        summary = response.strip()
    
    return summary

# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    example_code = """
    def factorial(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * factorial(n-1)
    """
    
    def simple_loading_animation(elapsed):
        symbols = ['-', '\\', '|', '/']
        idx = int(elapsed * 5) % len(symbols)
        print(f"\rGenerating summary... {symbols[idx]}", end="")
    
    print("Testing code summarization:")
    summary = get_code_summary(example_code, loading_callback=simple_loading_animation)
    print("\nSummary:", summary)
