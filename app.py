import streamlit as st
import os
from test import process_cpp_repository, save_dict_to_json, read_json_and_list_functions

st.title("C++ Code Repository Analyzer")

# Input box for repository path
repo_path = st.text_input("Enter the path to your C++ repository:", "")

# Index button
if st.button("Index"):
    if repo_path and os.path.exists(repo_path):
        st.info("Processing repository... This may take a while.")
        try:
            # Process the repository
            results = process_cpp_repository(repo_path)
            
            # Save results to JSON in current directory
            output_file = "cpp_analysis_results.json"
            save_dict_to_json(results, output_file)
            
            st.success(f"Repository analysis completed! Results saved to {output_file}")
            
            # Display some basic statistics
            st.subheader("Analysis Summary:")
            st.write(f"- Files processed: {len(results['file_results'])}")
            st.write(f"- Files with errors: {len(results['error_files'])}")
            
            if results['error_files']:
                st.error("Files that failed processing:")
                for error_file in results['error_files']:
                    st.write(f"- {error_file['file']}: {error_file['error']}")
                    
        except Exception as e:
            st.error(f"An error occurred while processing the repository: {str(e)}")
    else:
        st.error("Please enter a valid repository path.")


st.header("Read Analysis Results from JSON")
json_file_path = st.text_input("Enter the path to the JSON file:", "")

if st.button("Load and Display Functions"):
    if json_file_path and os.path.exists(json_file_path):
        st.info("Reading JSON file and extracting function information...")
        try:
            # Read the JSON file and extract file-function mappings
            file_function_dict = read_json_and_list_functions(json_file_path)
            
            # Display the results in collapsible sections
            st.subheader("File Names and Their Functions:")
            for file_name, function_details in file_function_dict.items():
                with st.expander(f"📁 {file_name} ({len(function_details)} functions)"):
                    if function_details:
                        for func in function_details:
                            st.markdown(f"**Function Name:** `{func['name']}`")
                            st.markdown(f"- **Signature:** `{func['signature']}`")
                            st.markdown(f"- **Lines:** {func['line_start'] + 1} - {func['line_end'] + 1}")
                            if func['comments']:
                                st.markdown("- **Comments:**")
                                for comment in func['comments']:
                                    st.markdown(f"  - `{comment}`")
                    else:
                        st.write("No functions found in this file.")
        except Exception as e:
            st.error(f"An error occurred while loading or displaying the JSON data: {e}")
    else:
        st.error("Please enter a valid JSON file path.")
