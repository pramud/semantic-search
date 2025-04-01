from tree_sitter import Language, Parser
import os
import json
from collections import defaultdict

def build_call_adjacency(call_graph):
    graph = defaultdict(list)
    for edge in call_graph:
        graph[edge['from']].append(edge['to'])
    return graph


languages_lib_path = "../build/languages.dll"
language = Language(languages_lib_path, 'cpp')
parser = Parser()
parser.set_language(language)

def parse_cpp_code(code: str):
    try:
        tree = parser.parse(bytes(code, 'utf-8'))
        return tree.root_node
    except Exception as e:
        print(f"Error parsing code: {e}")
        return None

def get_node_text(node, code: str):
    return code[node.start_byte:node.end_byte] if node else ""

def find_child_by_type(node, node_type: str):
    if node.type == node_type:
        return node
    for child in node.children:
        result = find_child_by_type(child, node_type)
        if result:
            return result
    return None

def extract_specifiers(node, code: str):
    specifiers = []
    for child in node.children:
        if child.type in ['storage_class_specifier', 'type_qualifier', 'function_specifier']:
            specifiers.append(get_node_text(child, code))
        elif child.type in ['function_declarator', 'type_identifier']:
            break
    return specifiers

import re

def extract_function_calls(node, code: str):
    calls = []

    if node.type == 'call_expression':
        func_node = node.child_by_field_name('function')
        if func_node:
            raw_text = get_node_text(func_node, code).strip()
            
            # Clean up extra junk from chained expressions
            clean_name = re.sub(r'[^a-zA-Z0-9_:~><]+$', '', raw_text)  # remove trailing junk
            if clean_name:
                calls.append(clean_name)

    for child in node.children:
        calls += extract_function_calls(child, code)
    return calls


def extract_used_identifiers(node, code: str):
    identifiers = []
    if node.type == 'identifier':
        identifiers.append(get_node_text(node, code))
    for child in node.children:
        identifiers += extract_used_identifiers(child, code)
    return identifiers

def extract_adjacent_comments(node, code: str):
    comments = []
    current_node = node.prev_sibling
    while current_node and current_node.type in ['comment', 'preproc']:
        comments.append(get_node_text(current_node, code))
        current_node = current_node.prev_sibling
    return comments[::-1]

def get_function_signature(node, code: str):
    declarator = find_child_by_type(node, 'function_declarator')
    if not declarator:
        return ""
    return_type_node = None
    current = declarator.prev_sibling
    while current:
        if current.type in ['type_identifier', 'auto', 'decltype']:
            return_type_node = current
            break
        current = current.prev_sibling
    return_type = get_node_text(return_type_node, code) if return_type_node else 'void'
    parameters = get_node_text(find_child_by_type(declarator, 'parameter_list'), code)
    return f"{return_type} {get_node_text(declarator.children[0], code)} {parameters}"

def get_scope(node):
    scopes = []
    parent = node.parent
    while parent:
        name_node = None

        if parent.type == 'namespace_definition':
            name_node = find_child_by_type(parent, 'identifier')
            if name_node:
                scopes.insert(0, name_node.text.decode())
            else:
                scopes.insert(0, '(anonymous namespace)')

        elif parent.type in ['class_specifier', 'struct_specifier']:
            name_node = find_child_by_type(parent, 'type_identifier')
            if name_node:
                scopes.insert(0, name_node.text.decode())

        parent = parent.parent

    return "::".join(scopes)


def extract_functions(node, code: str, file_name: str):
    functions = []
    if node.type == 'function_definition' or node.type == 'template_declaration':
        is_template = node.type == 'template_declaration'
        func_node = node if not is_template else find_child_by_type(node, 'function_definition')
        if not func_node:
            return functions

        declarator_node = find_child_by_type(func_node, 'function_declarator')
        name_parts = get_node_text(declarator_node, code).split('(')[0].strip().split('::')
        normalized_name = name_parts[-1]
        is_operator = normalized_name.startswith('operator')
        body_node = find_child_by_type(func_node, 'compound_statement')

        calls = extract_function_calls(body_node, code) if body_node else []
        identifiers = extract_used_identifiers(body_node, code) if body_node else []
        specifiers = extract_specifiers(func_node, code)

        # Determine scope from declarator (handles out-of-line definitions like ClassName::Func)
        declarator_text = get_node_text(declarator_node, code).split('(')[0].strip()
        declarator_parts = declarator_text.split('::')
        normalized_name = declarator_parts[-1]
        scope_from_declarator = '::'.join(declarator_parts[:-1])

        # Determine scope from AST tree (for class inside namespace, etc.)
        scope_from_tree = get_scope(func_node)

        # Merge both scopes if they differ (e.g., namespace + class)
        if scope_from_declarator and scope_from_tree and not scope_from_tree.endswith(scope_from_declarator):
            full_scope = f"{scope_from_tree}::{scope_from_declarator}"
        elif scope_from_declarator:
            full_scope = scope_from_declarator
        else:
            full_scope = scope_from_tree

        qualified_name = f"{full_scope}::{normalized_name}" if full_scope else normalized_name

        functions.append({
            'type': 'function',
            'name': normalized_name,
            'qualified_name': qualified_name,
            'signature': get_function_signature(func_node, code),
            'template': is_template,
            'specifiers': specifiers,
            'calls': calls,
            'used_identifiers': identifiers,
            'body': get_node_text(func_node, code),
            'line_start': func_node.start_point[0],
            'line_end': func_node.end_point[0],
            'comments': extract_adjacent_comments(func_node, code),
            'file': file_name
        })
    for child in node.children:
        functions += extract_functions(child, code, file_name)
    return functions

def build_call_graph(functions):
    edges = []
    name_map = {f['name']: f['qualified_name'] for f in functions}

    for func in functions:
        for call in func['calls']:
            cleaned_call = call.strip()

            target = name_map.get(cleaned_call, cleaned_call)

            edges.append({
                'from': func['qualified_name'],
                'from_file': func['file'],
                'from_line': func['line_start'],
                'to': target,
                'call_text': cleaned_call
            })
    return edges


def get_cpp_files(root_dir):
    cpp_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
                cpp_files.append(os.path.join(dirpath, filename))
    return cpp_files

def process_files(file_paths):
    all_functions = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = parse_cpp_code(content)
            functions = extract_functions(tree, content, file_path)
            all_functions.extend(functions)
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
    return all_functions

def find_all_flows(graph):
    flows = []

    def dfs(path):
        current = path[-1]
        if current not in graph or not graph[current]:  # end of flow
            flows.append(path)
            return
        for neighbor in graph[current]:
            if neighbor not in path:  # avoid cycles
                dfs(path + [neighbor])

    # Start from each function (entry point)
    for start in graph.keys():
        dfs([start])

    return flows


def parse_directory(root_dir: str):
    """Parses C++ files in a directory, extracts functions, builds call graph and flows."""
    # Initialize tree-sitter here to ensure it's set up when called externally
    languages_lib_path = "../build/languages.dll"
    if not os.path.exists(languages_lib_path):
        raise FileNotFoundError(f"Tree-sitter language library not found at {languages_lib_path}")
    language = Language(languages_lib_path, 'cpp')
    parser = Parser()
    parser.set_language(language)

    # Ensure parser is passed or available to functions needing it
    # (Note: Current implementation uses global parser, which might need adjustment if run concurrently)
    
    cpp_files = get_cpp_files(root_dir)
    if not cpp_files:
        print(f"No C++ files found in {root_dir}")
        return None # Or raise an error, or return empty structure

    all_functions = process_files(cpp_files) # Assuming process_files uses the global parser
    call_graph = build_call_graph(all_functions)
    adj_graph = build_call_adjacency(call_graph)
    flows = find_all_flows(adj_graph)
    output = {
        "functions": all_functions,
        "call_graph": call_graph,
        "flows": flows
    }
    return output

# ---- Run Parser on a File ----
if __name__ == "__main__":
    root_dir_arg = "./test_project"  # Or wherever your repo is
    parsed_data = parse_directory(root_dir_arg)
    if parsed_data:
        print(json.dumps(parsed_data, indent=2))
    else:
        print(f"Parsing failed for directory: {root_dir_arg}")
