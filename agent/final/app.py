import streamlit as st
import os
import json
import sys
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Qwen model (do this once at startup)
@st.cache_resource
def load_qwen_model():
    try:
        model_name = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Qwen model: {e}")
        return None, None

def get_qwen_analysis(model, tokenizer, text, instruction):
    try:
        prompt = f"<|im_start|>user\n{instruction}\nHere's the code and context:\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        return response
    except Exception as e:
        return f"Error generating analysis: {e}"

def get_function_details(parsed_data, function_name):
    """Get all details about a function and its call stack from the parsed data."""
    functions = parsed_data.get("functions", [])
    call_graph = parsed_data.get("call_graph", [])
    
    # Find the target function
    target_func = None
    for func in functions:
        if func["qualified_name"] == function_name:
            target_func = func
            break
    
    if not target_func:
        return None, [], []
    
    # Get functions called by this function (direct calls)
    called_functions = []
    for edge in call_graph:
        if edge["from"] == function_name:
            called_func = next((f for f in functions if f["qualified_name"] == edge["to"]), None)
            if called_func:
                called_functions.append(called_func)
    
    # Get functions that call this function (callers)
    calling_functions = []
    for edge in call_graph:
        if edge["to"] == function_name:
            caller_func = next((f for f in functions if f["qualified_name"] == edge["from"]), None)
            if caller_func:
                calling_functions.append(caller_func)
    
    return target_func, called_functions, calling_functions

# --- Helper Function to Add Directory to Path ---
def add_path(path_to_add):
    """Adds a directory to the Python path if it's not already there."""
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)
        logging.info(f"Added {path_to_add} to sys.path")

# --- Import Parsing Logic ---
# Assume final/parse_files.py exists relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
final_module_dir = os.path.join(script_dir, 'final')

# Add the 'final' directory to sys.path to allow direct import
add_path(final_module_dir)
# Also add the parent directory in case the script is run from elsewhere
add_path(script_dir) 

try:
    # Attempt to import the parsing function
    from parse_files import parse_directory
    logging.info("Successfully imported 'parse_directory'.")
except ImportError as e:
    logging.error(f"ImportError: {e}. Check if 'final/parse_files.py' exists and is accessible.")
    error_msg = (
        f"Fatal Error: Could not import the parsing logic from 'final/parse_files.py'.\n"
        f"Please ensure the file exists relative to the app and there are no errors in it.\n"
        f"Import error: {e}\n"
        f"Script directory: {script_dir}\n"
        f"Attempted module directory: {final_module_dir}\n"
        f"Current sys.path: {sys.path}"
    )
    st.error(error_msg)
    st.stop()
except Exception as e:
    logging.error(f"An unexpected error occurred during import: {e}", exc_info=True)
    st.error(f"An unexpected error occurred trying to load the parsing module: {e}")
    st.stop()


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("C++ Codebase Parser")

# Initialize tabs
tab1, tab2 = st.tabs(["Parse Code", "Analyze Functions"])

with tab1:
    st.write("Select a directory containing C++ source code and specify an output filename. "
             "The app will parse the files, analyze function calls, and save the results to a JSON file in the `./parsed` directory.")
    
    # Input Fields for parsing
    default_dir = "./test_project"
    if not os.path.isdir(default_dir):
        st.warning(f"Default directory '{default_dir}' not found. Please specify a valid directory.")
        default_dir = "."

    root_dir_input = st.text_input("Directory to Parse:", value=default_dir, 
                                  help="Path to the root directory containing C++ files.")
    output_filename_input = st.text_input("Output JSON Filename:", value="parsed_code_data.json",
                                        help="The name for the output JSON file (will be saved in ./parsed).")

    # Parse Button and logic
    if st.button("Parse Directory", type="primary"):
        # --- Input Validation ---
        parse_dir = root_dir_input.strip()
        output_filename = output_filename_input.strip()

        valid_input = True
        if not parse_dir:
            st.error("Please provide a directory path to parse.")
            valid_input = False
        elif not os.path.isdir(parse_dir):
             st.error(f"Directory not found: '{parse_dir}'")
             valid_input = False

        if not output_filename:
            st.error("Please provide an output filename.")
            valid_input = False
        elif not output_filename.lower().endswith(".json"):
             output_filename += ".json"
             st.info(f"Appending '.json' to filename: {output_filename}")


        if valid_input:
            # --- Setup Output ---
            parsed_output_dir = os.path.join(script_dir, "parsed")
            output_path = os.path.join(parsed_output_dir, output_filename)
            logging.info(f"Target output path: {output_path}")

            try:
                # Create ./parsed directory if it doesn't exist
                os.makedirs(parsed_output_dir, exist_ok=True)
                logging.info(f"Ensured output directory exists: {parsed_output_dir}")

                # --- Run Parsing ---
                with st.spinner(f"Parsing C++ files in '{parse_dir}'... This may take a while."):
                    logging.info(f"Starting parsing for directory: {parse_dir}")
                    parsed_data = parse_directory(parse_dir) # Call the imported function
                    logging.info(f"Parsing finished for directory: {parse_dir}")

                if parsed_data is None:
                     st.warning(f"Parsing completed, but no data was generated. This could mean no C++ files were found in '{parse_dir}' or there was an issue during parsing. Check logs if available.")
                     logging.warning(f"parse_directory returned None for {parse_dir}")
                else:
                    # --- Save Output ---
                    logging.info(f"Attempting to save parsed data to {output_path}")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_data, f, indent=2)
                    logging.info(f"Successfully saved data to {output_path}")
                    st.success(f"Successfully parsed directory and saved results to:")
                    st.code(output_path, language=None)

                    # --- Display Results (Optional Preview) ---
                    st.subheader("Parsed Data Preview (Functions)")
                    if parsed_data.get("functions"):
                        st.dataframe(parsed_data["functions"][:20]) # Show first 20 functions
                        if len(parsed_data["functions"]) > 20:
                            st.caption(f"... and {len(parsed_data['functions']) - 20} more functions.")
                    else:
                        st.write("No function data found.")

                    # Add more previews if desired (e.g., call graph edges)


            except FileNotFoundError as e:
                st.error(f"Error during parsing: {e}. This often means the tree-sitter language library (e.g., './build/languages.dll' relative to the parser script) was not found.")
                logging.error(f"FileNotFoundError during parsing: {e}", exc_info=True)
            except ImportError as e: # Catch potential import errors within the called function
                 st.error(f"Import Error within parsing logic: {e}. Ensure all dependencies for `parse_files.py` are installed.")
                 logging.error(f"ImportError during parse_directory call: {e}", exc_info=True)
            except Exception as e:
                st.error(f"An unexpected error occurred during parsing or saving:")
                st.exception(e) # Show full traceback in the app
                logging.error(f"Unexpected error: {e}", exc_info=True)

with tab2:
    st.write("Select a parsed JSON file and analyze specific functions using the Qwen model.")
    
    # Get list of JSON files in the parsed directory
    parsed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsed")
    if os.path.exists(parsed_dir):
        json_files = [f for f in os.listdir(parsed_dir) if f.endswith('.json')]
        
        if json_files:
            # File selection
            selected_file = st.selectbox("Select Parsed JSON File:", json_files)
            
            try:
                with open(os.path.join(parsed_dir, selected_file), 'r') as f:
                    parsed_data = json.load(f)
                
                # Get list of functions
                functions = parsed_data.get("functions", [])
                if functions:
                    function_names = [f["qualified_name"] for f in functions]
                    selected_function = st.selectbox("Select Function to Analyze:", function_names)
                    
                    if selected_function:
                        # Load model when needed
                        model, tokenizer = load_qwen_model()
                        
                        if model and tokenizer:
                            target_func, called_funcs, calling_funcs = get_function_details(parsed_data, selected_function)
                            
                            if target_func:
                                with st.spinner("Analyzing function..."):
                                    # Function summary
                                    st.subheader("Function Analysis")
                                    summary_prompt = f"Analyze this C++ function and provide a detailed summary of its purpose, parameters, return value, and key functionality:\n{target_func['body']}"
                                    function_summary = get_qwen_analysis(model, tokenizer, target_func['body'], summary_prompt)
                                    st.write(function_summary)
                                    
                                    # Call stack analysis
                                    st.subheader("Call Stack Analysis")
                                    
                                    # Called functions
                                    if called_funcs:
                                        st.write("### Functions Called by This Function")
                                        called_funcs_text = "\n".join([f"{f['qualified_name']}:\n{f['body']}" for f in called_funcs])
                                        call_stack_prompt = "Analyze these functions that are called by the main function. Explain how they work together and their dependencies:"
                                        call_stack_analysis = get_qwen_analysis(model, tokenizer, called_funcs_text, call_stack_prompt)
                                        st.write(call_stack_analysis)
                                    
                                    # Calling functions
                                    if calling_funcs:
                                        st.write("### Functions That Call This Function")
                                        calling_funcs_text = "\n".join([f"{f['qualified_name']}:\n{f['body']}" for f in calling_funcs])
                                        callers_prompt = "Analyze these functions that call the main function. Explain the context in which the main function is used:"
                                        callers_analysis = get_qwen_analysis(model, tokenizer, calling_funcs_text, callers_prompt)
                                        st.write(callers_analysis)
                                    
                                    # Overall call stack summary
                                    st.subheader("Call Stack Summary")
                                    stack_summary_prompt = f"Provide a concise summary of the entire call stack, showing how data flows between these functions:\nMain function: {target_func['qualified_name']}\nCalls: {[f['qualified_name'] for f in called_funcs]}\nCalled by: {[f['qualified_name'] for f in calling_funcs]}"
                                    stack_summary = get_qwen_analysis(model, tokenizer, stack_summary_prompt, "Summarize this call stack flow:")
                                    st.write(stack_summary)
                            else:
                                st.error("Could not find detailed information about the selected function.")
                        else:
                            st.error("Failed to load the Qwen model. Please check your installation and try again.")
                else:
                    st.warning("No functions found in the selected file.")
            except Exception as e:
                st.error(f"Error loading or processing the JSON file: {e}")
        else:
            st.warning("No parsed JSON files found. Please parse a C++ codebase first.")
    else:
        st.warning("No parsed directory found. Please parse a C++ codebase first.") 
