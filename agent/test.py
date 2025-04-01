from tree_sitter import Language, Parser
import os
import json

languages_lib_path = "./build/languages.dll"
language = Language(languages_lib_path, 'cpp')
parser = Parser()
parser.set_language(language)
language = Language(languages_lib_path, 'cpp')


def parse_cpp_code(code: str):
    """Parse C++ code into a Tree-sitter syntax tree."""
    try:
        tree = parser.parse(bytes(code, 'utf-8'))
        return tree.root_node
    except Exception as e:
        print(f"Error parsing code: {e}")
        return None

def extract_functions(node, code: str):
    """Extract function definitions with templates, operator overloading, and specializations."""
    functions = []
    if node.type == 'template_declaration':
        params_node = find_child_by_type(node, 'template_parameter_list')
        params = get_node_text(params_node, code)
        is_specialization = params == '<>'
        func_node = find_child_by_type(node, 'function_definition')
        if func_node:
            name_node = find_child_by_type(func_node, 'function_declarator')
            name = get_node_text(name_node, code)
            # Extract normalized name
            normalized_name = name.split('(')[0].strip()
            is_operator = name.startswith('operator')
            body_node = find_child_by_type(func_node, 'compound_statement')
            specifiers = extract_specifiers(func_node, code)
            calls = extract_function_calls(body_node, code) if body_node else []
            identifiers = extract_used_identifiers(body_node, code) if body_node else []
            functions.append({
                'type': 'function',
                'name': name,
                'normalized_name': normalized_name,  # Added normalized name
                'signature': get_function_signature(func_node, code),
                'template_params': params,
                'is_specialization': is_specialization,
                'is_operator': is_operator,
                'specifiers': specifiers,
                'body': get_node_text(func_node, code),
                'calls': calls,
                'used_identifiers': identifiers,
                'line_start': func_node.start_point[0],
                'line_end': func_node.end_point[0],
                'comments': extract_adjacent_comments(func_node, code)
            })
    elif node.type == 'function_definition':
        name_node = find_child_by_type(node, 'function_declarator')
        name = get_node_text(name_node, code)
        # Extract normalized name
        normalized_name = name.split('(')[0].strip()
        is_operator = name.startswith('operator')
        body_node = find_child_by_type(node, 'compound_statement')
        specifiers = extract_specifiers(node, code)
        calls = extract_function_calls(body_node, code) if body_node else []
        identifiers = extract_used_identifiers(body_node, code) if body_node else []
        functions.append({
            'type': 'function',
            'name': name,
            'normalized_name': normalized_name,  # Added normalized name
            'signature': get_function_signature(node, code),
            'template_params': '',
            'is_specialization': False,
            'is_operator': is_operator,
            'specifiers': specifiers,
            'body': get_node_text(node, code),
            'calls': calls,
            'used_identifiers': identifiers,
            'line_start': node.start_point[0],
            'line_end': node.end_point[0],
            'comments': extract_adjacent_comments(node, code)
        })
    for child in node.children:
        functions += extract_functions(child, code)
    return functions

def extract_function_calls(node, code: str):
    """Extract function calls within a node (e.g., function body)."""
    calls = []
    if node.type == 'call_expression':
        func_name_node = node.child_by_field_name('function')
        if func_name_node:
            calls.append(get_node_text(func_name_node, code))
    for child in node.children:
        calls += extract_function_calls(child, code)
    return calls

def extract_used_identifiers(node, code: str):
    """Extract all identifiers used within a node."""
    identifiers = []
    if node.type == 'identifier':
        identifiers.append(get_node_text(node, code))
    for child in node.children:
        identifiers += extract_used_identifiers(child, code)
    return identifiers

def get_function_signature(node, code: str) -> str:
    """Extract function signature (return type + parameters)."""
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



def find_child_by_type(node, node_type: str):
    """Find first child of specific type"""
    if node.type == node_type:
        return node
    for child in node.children:
        result = find_child_by_type(child, node_type)
        if result: return result
    return None

def get_node_text(node, code: str):
    """Get the text content of a node."""
    if node:
        return code[node.start_byte:node.end_byte]
    return ""

def extract_adjacent_comments(node, code: str):
    """Extract comments before/after node"""
    comments = []
    current_node = node.prev_sibling
    while current_node and current_node.type in ['comment', 'preproc']:
        comments.append(get_node_text(current_node, code))
        current_node = current_node.prev_sibling
    return comments[::-1]  # Reverse to preserve order

def extract_specifiers(node, code: str):
    """Extract specifiers like constexpr, static, etc., from a function definition."""
    specifiers = []
    for child in node.children:
        if child.type in ['storage_class_specifier', 'type_qualifier', 'function_specifier']:
            specifiers.append(get_node_text(child, code))
        elif child.type in ['function_declarator', 'type_identifier']:
            break  # Specifiers come before the declarator or type
    return specifiers

def get_function_signature(node, code: str) -> str:
    """Extract function signature (return type + parameters)"""
    declarator = find_child_by_type(node, 'function_declarator')
    if not declarator:
        return ""
    
    # Get return type (could be type_identifier or more complex)
    return_type_node = None
    current = declarator.prev_sibling
    while current:
        if current.type in ['type_identifier', 'auto', 'decltype']:
            return_type_node = current
            break
        current = current.prev_sibling
    
    return_type = get_node_text(return_type_node, code) if return_type_node else 'void'
    parameters = get_node_text(find_child_by_type(declarator, 'parameter_list'), code)
    
    return f"{return_type} {declarator.children[0].text.decode()} {parameters}"

def process_cpp_file(file_content: str, file_path: str = None):
    macro_regions = preprocess_macros(file_content)
    root_node = parse_cpp_code(file_content)
    if not root_node:
        return {'file_path': file_path, 'error': 'Failed to parse file'}

    macros = extract_macros(root_node, file_content)
    macro_names = {m['name'] for m in macros}
    macro_usages = find_macro_usages(root_node, file_content, macro_names)
    modern_features = extract_modern_cpp_features(root_node, file_content)

    result = {
        'file_path': file_path,
        'functions': extract_functions(root_node, file_content),
        'classes': extract_classes(root_node, file_content),
        'includes': extract_includes(root_node, file_content),
        'macros': macros,
        'macro_usages': macro_usages,
        'enums': extract_enums(root_node, file_content),
        'namespaces': extract_namespaces(root_node, file_content),
        'lambdas': extract_lambdas(root_node, file_content),
        'template_instantiations': extract_template_instantiations(root_node, file_content),
        'concepts': extract_concepts(root_node, file_content),
        'variable_references': extract_variable_references(root_node, file_content),
        'modern_features': modern_features,
        'macro_regions': macro_regions
    }

    result['dependency_graph'] = build_dependency_graph(result)
    return result



def extract_includes(node, code: str):
    """Extract #include directives."""
    includes = []
    if node.type == 'preproc_include':
        path_node = find_child_by_type(node, 'system_lib_string')
        includes.append({
            'path': get_node_text(path_node, code).strip('<>"'),
            'line': node.start_point[0]
        })
    for child in node.children:
        includes += extract_includes(child, code)
    return includes

def extract_enums(node, code: str):
    """Extract enums and enum classes."""
    enums = []
    if node.type == 'enum_specifier':
        enum_name = find_child_by_type(node, 'type_identifier')
        underlying_type = find_child_by_type(node, 'enum_base')
        enumerators = []
        for child in node.children:
            if child.type == 'enumerator':
                name_node = find_child_by_type(child, 'identifier')
                value_node = find_child_by_type(child, 'number_literal')  # Simplified
                enumerators.append({
                    'name': get_node_text(name_node, code),
                    'value': get_node_text(value_node, code) if value_node else ''
                })
        enums.append({
            'name': get_node_text(enum_name, code) if enum_name else '<anonymous>',
            'scoped': 'class' in get_node_text(node, code),
            'underlying_type': get_node_text(underlying_type, code) if underlying_type else 'int',
            'enumerators': enumerators,
            'line_start': node.start_point[0],
            'line_end': node.end_point[0]
        })
    for child in node.children:
        enums += extract_enums(child, code)
    return enums

def extract_namespaces(node, code: str):
    """Extract namespace definitions."""
    namespaces = []
    if node.type == 'namespace_definition':
        name_node = find_child_by_type(node, 'identifier')
        body_node = find_child_by_type(node, 'compound_statement')  # Adjusted type
        nested = extract_namespaces(body_node, code) if body_node else []
        namespaces.append({
            'name': get_node_text(name_node, code) if name_node else '<anonymous>',
            'line_start': node.start_point[0],
            'line_end': node.end_point[0],
            'nested_namespaces': nested,
            'declarations': extract_namespace_declarations(body_node, code)
        })
    for child in node.children:
        namespaces += extract_namespaces(child, code)
    return namespaces

def extract_namespace_declarations(body_node, code: str):
    """Extract declarations within a namespace."""
    declarations = {
        'functions': [],
        'classes': [],
        'variables': [],
        'enums': [],
        'using_directives': []
    }
    if not body_node:
        return declarations
    for child in body_node.children:
        if child.type == 'function_definition':
            declarations['functions'].extend(extract_functions(child, code))
        elif child.type in ['class_specifier', 'struct_specifier']:
            declarations['classes'].extend(extract_classes(child, code))
        elif child.type == 'declaration':
            declarator = find_child_by_type(child, 'init_declarator')
            if declarator:
                declarations['variables'].append({
                    'name': get_node_text(declarator, code),
                    'type': get_node_text(find_child_by_type(child, 'type_identifier'), code)
                })
        elif child.type == 'enum_specifier':
            declarations['enums'].extend(extract_enums(child, code))
        elif child.type == 'using_declaration':
            declarations['using_directives'].append(get_node_text(child, code))
    return declarations

def get_access_modifier(node):
    """Determine access modifier (public/private/protected)."""
    current = node.prev_sibling
    while current:
        if current.type == 'access_specifier':
            return get_node_text(current, code).rstrip(':')
        current = current.prev_sibling
    return 'private'  # Default for classes


def extract_macros(node, code: str):
    """Extract #define macros."""
    macros = []
    if node.type == 'preproc_def':
        macro_name = find_child_by_type(node, 'identifier')
        value_node = find_child_by_type(node, 'preproc_arg')
        macros.append({
            'name': get_node_text(macro_name, code),
            'value': get_node_text(value_node, code) if value_node else '',
            'line': node.start_point[0]
        })
    for child in node.children:
        macros += extract_macros(child, code)
    return macros

def find_macro_usages(root_node, code: str, macro_names):
    """Find potential macro usages by matching identifiers."""
    usages = []
    def traverse(node):
        if node.type == 'identifier' and get_node_text(node, code) in macro_names:
            usages.append({
                'macro': get_node_text(node, code),
                'line': node.start_point[0]
            })
        for child in node.children:
            traverse(child)
    traverse(root_node)
    return usages


def extract_classes(node, code: str):
    """Extract classes/structs with templates, specializations, and friend declarations."""
    classes = []
    if node.type == 'template_declaration':
        params_node = find_child_by_type(node, 'template_parameter_list')
        params = get_node_text(params_node, code)
        is_specialization = params == '<>'
        class_node = find_child_by_type(node, 'class_specifier') or find_child_by_type(node, 'struct_specifier')
        if class_node:
            class_name = find_child_by_type(class_node, 'type_identifier')
            members = extract_class_members(class_node, code)
            methods = extract_class_methods(class_node, code)
            classes.append({
                'type': class_node.type,
                'name': get_node_text(class_name, code),
                'template_params': params,
                'is_specialization': is_specialization,
                'inherits': extract_base_classes(class_node, code),  # Correct here
                'members': members,
                'methods': methods,
                'line_start': class_node.start_point[0],
                'line_end': class_node.end_point[0]
            })
    elif node.type in ['class_specifier', 'struct_specifier']:
        class_name = find_child_by_type(node, 'type_identifier')
        members = extract_class_members(node, code)
        methods = extract_class_methods(node, code)
        classes.append({
            'type': node.type,
            'name': get_node_text(class_name, code),
            'template_params': '',
            'is_specialization': False,
            'inherits': extract_base_classes(node, code),  # Fix: Use node instead of class_node
            'members': members,
            'methods': methods,
            'line_start': node.start_point[0],
            'line_end': node.end_point[0]
        })
    for child in node.children:
        classes += extract_classes(child, code)
    return classes


def extract_class_members(node, code: str):
    """Extract class members, including friend declarations."""
    members = []
    for child in node.children:
        if child.type == 'field_declaration':
            declarator = find_child_by_type(child, 'field_identifier')
            type_node = find_child_by_type(child, 'type_identifier')
            members.append({
                'type': 'member',
                'name': get_node_text(declarator, code),
                'data_type': get_node_text(type_node, code) if type_node else 'auto',
                'access': get_access_modifier(child),
                'line': child.start_point[0]
            })
        elif child.type == 'friend_declaration':
            friend_text = get_node_text(child, code)
            members.append({
                'type': 'friend',
                'declaration': friend_text,
                'line': child.start_point[0]
            })
    return members

def extract_template_instantiations(node, code: str):
    """Extract template instantiations (e.g., std::vector<int>)."""
    
    instantiations = []

    # Forward declarations
    if node.type == 'template_declaration':
        if class_node := find_child_by_type(node, ['class_specifier', 'struct_specifier']):
            name_node = find_child_by_type(class_node, 'type_identifier')
            params = get_node_text(find_child_by_type(node, 'template_parameter_list'), code)
            instantiations.append({
                'type': 'forward_declaration',
                'name': get_node_text(name_node, code),
                'parameters': params,
                'line': node.start_point[0]
            })
    # Alias templates
    if node.type == 'alias_declaration':
        if template_node := find_child_by_type(node, 'template_type'):
            name = get_node_text(find_child_by_type(node, 'identifier'), code)
            target = get_node_text(template_node, code)
            instantiations.append({
                'type': 'alias',
                'name': name,
                'target': target,
                'line': node.start_point[0]
            })

    if node.type == 'template_type':
        template_name = get_node_text(node.child_by_field_name('name'), code)
        arguments = [get_node_text(arg, code) for arg in node.child_by_field_name('arguments').children if arg.type == 'type_identifier']
        instantiations.append({
            'template': template_name,
            'arguments': arguments,
            'line': node.start_point[0]
        })
    for child in node.children:
        instantiations += extract_template_instantiations(child, code)
    return instantiations

def preprocess_macros(code: str):
    """Track macro expansion regions and conditional compilation"""
    macro_regions = []
    current_region = []
    in_conditional = False
    
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#ifdef') or stripped.startswith('#ifndef'):
            in_conditional = True
            current_region = [stripped]
        elif stripped.startswith('#endif'):
            if in_conditional:
                current_region.append(stripped)
                macro_regions.append('\n'.join(current_region))
                current_region = []
                in_conditional = False
        elif in_conditional:
            current_region.append(line)
    
    return macro_regions

def extract_modern_cpp_features(node, code: str):
    """Extract C++20 modules, coroutines, and ranges"""
    features = {
        'modules': [],
        'coroutines': [],
        'ranges': []
    }
    
    if node.type == 'import_declaration':
        features['modules'].append({
            'module': get_node_text(node, code),
            'line': node.start_point[0]
        })
    
    if node.type in ['co_await_expression', 'co_yield_expression']:
        features['coroutines'].append({
            'type': node.type,
            'line': node.start_point[0]
        })
    
    if node.type == 'range_based_for_statement':
        features['ranges'].append({
            'declaration': get_node_text(node, code),
            'line': node.start_point[0]
        })
    
    for child in node.children:
        child_features = extract_modern_cpp_features(child, code)
        features['modules'] += child_features['modules']
        features['coroutines'] += child_features['coroutines']
        features['ranges'] += child_features['ranges']
    
    return features


def extract_concepts(node, code: str):
    """Extract C++20 concept definitions."""
    concepts = []
    if node.type == 'concept_definition':
        name_node = find_child_by_type(node, 'identifier')
        concepts.append({
            'name': get_node_text(name_node, code),
            'line_start': node.start_point[0],
            'line_end': node.end_point[0]
        })
    for child in node.children:
        concepts += extract_concepts(child, code)
    return concepts

def extract_variable_references(node, code: str):
    """Track variable references with scope hierarchy"""
    references = {}
    scope_stack = [('global', 0)]
    
    def traverse(node):
        nonlocal scope_stack
        
        if node.type == 'function_definition':
            scope_stack.append(('function', node.start_point[0]))
        elif node.type == 'compound_statement':
            scope_stack.append(('block', node.start_point[0]))
        
        if node.type == 'declaration':
            # Track variable definitions
            declarator = find_child_by_type(node, 'init_declarator')
            if declarator and (name_node := find_child_by_type(declarator, 'identifier')):
                var_name = get_node_text(name_node, code)
                scope_key = '.'.join([f"{s[0]}:{s[1]}" for s in scope_stack])
                references[var_name] = references.get(var_name, []) + [{
                    'scope': scope_key,
                    'type': 'definition',
                    'line': node.start_point[0]
                }]
        
        elif node.type == 'identifier':
            var_name = get_node_text(node, code)
            if var_name in references:
                scope_key = '.'.join([f"{s[0]}:{s[1]}" for s in scope_stack])
                references[var_name].append({
                    'scope': scope_key,
                    'type': 'reference',
                    'line': node.start_point[0]
                })
        
        for child in node.children:
            traverse(child)
        
        if node.type in ['function_definition', 'compound_statement']:
            scope_stack.pop()
    
    traverse(node)
    return references

def build_dependency_graph(data):
    """Create relationship graphs from extracted data"""
    graph = {
        'includes': {},
        'inheritance': {},
        'calls': {},
        'overrides': {}
    }
    
    # Include dependencies
    for include in data['includes']:
        graph['includes'][include['path']] = graph['includes'].get(include['path'], []) + [data['file_path']]
    
    # Inheritance relationships
    for cls in data['classes']:
        for base in cls['inherits']:
            graph['inheritance'].setdefault(base, []).append(cls['name'])
    
    # Function call graph
    for func in data['functions']:
        for call in func['calls']:
            graph['calls'].setdefault(func['name'], []).append(call)
    
    # Method overrides
    for cls in data['classes']:
        for method in cls['methods']:
            if 'override' in method.get('specifiers', []):
                graph['overrides'].setdefault(cls['name'], []).append(method['name'])
    
    return graph


def extract_class_methods(node, code: str):
    """Extract methods from class/struct."""
    methods = []
    for child in node.children:
        if child.type in ['function_definition', 'constructor_definition']:
            methods.extend(extract_functions(child, code))
    return methods

def extract_base_classes(node, code: str):
    """Extract base classes from inheritance list."""
    base_clause = find_child_by_type(node, 'base_class_clause')  # Corrected type
    if not base_clause:
        return []
    return [get_node_text(base, code) for base in base_clause.children if base.type == 'type_identifier']

def extract_lambdas(node, code: str):
    """Extract lambda expressions."""
    lambdas = []
    if node.type == 'lambda_expression':
        capture_node = find_child_by_type(node, 'lambda_capture')
        params_node = find_child_by_type(node, 'parameter_list')
        body_node = find_child_by_type(node, 'compound_statement')
        lambdas.append({
            'type': 'lambda',
            'capture': get_node_text(capture_node, code),
            'parameters': get_node_text(params_node, code),
            'body': get_node_text(body_node, code),
            'line_start': node.start_point[0],
            'line_end': node.end_point[0]
        })
    for child in node.children:
        lambdas += extract_lambdas(child, code)
    return lambdas

def process_cpp_repository(repo_path: str):
    """
    Process all C++ files in a repository and return aggregated analysis results.
    
    Args:
        repo_path (str): Path to the C++ repository
        
    Returns:
        dict: Aggregated analysis results containing:
            - file_results (list): Per-file analysis data
            - global_dependency_graph (dict): Cross-file dependencies
            - error_files (list): Files that failed processing
    """
    results = {
        'file_results': [],
        'global_dependency_graph': {},
        'error_files': []
    }

    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if file_name.endswith((".cpp", ".h", ".hpp", ".cxx", ".hxx", ".cc", ".ipp")):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    file_result = process_cpp_file(file_content, file_path)
                    results['file_results'].append(file_result)
                    
                    # Aggregate dependency graph information
                    if 'dependency_graph' in file_result:
                        merge_dependency_graphs(
                            results['global_dependency_graph'],
                            file_result['dependency_graph']
                        )
                except Exception as e:
                    results['error_files'].append({
                        'file': file_path,
                        'error': str(e)
                    })
                    print(f"Error processing {file_path}: {str(e)}")

    return results

def save_dict_to_json(data_dict, file_path):
    """
    Save a Python dictionary to a JSON file.

    Parameters:
        data_dict (dict): The dictionary to be saved.
        file_path (str): The path where the JSON file will be saved.

    Returns:
        None
    """
    try:
        # Open the file in write mode ('w') and dump the dictionary into it
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)  # indent for pretty-printing
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary to JSON: {e}")

def merge_dependency_graphs(global_graph: dict, file_graph: dict):
    """Merge file-level dependency graph into global graph"""
    for key in file_graph:
        if key not in global_graph:
            global_graph[key] = {}
        for sub_key, values in file_graph[key].items():
            if sub_key not in global_graph[key]:
                global_graph[key][sub_key] = []
            global_graph[key][sub_key].extend(values)
            # Remove duplicates while preserving order
            global_graph[key][sub_key] = list(dict.fromkeys(global_graph[key][sub_key]))


def read_json_and_list_functions(json_file_path):
    """
    Reads a JSON file and lists all file names and their corresponding function details.
    
    Args:
        json_file_path (str): The path to the JSON file containing the analysis results.
    
    Returns:
        dict: A dictionary where keys are file names and values are lists of function details.
    """
    try:
        # Step 1: Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Step 2: Extract file names and function details
        file_function_dict = {}
        
        # Assuming the JSON structure has 'file_results' key which contains per-file analysis data
        for file_result in data.get('file_results', []):
            file_name = file_result.get('file_path', 'Unknown File')
            functions = file_result.get('functions', [])
            
            # Extract function details
            function_details = []
            for func in functions:
                function_details.append({
                    'name': func.get('name', 'Unnamed Function'),
                    'signature': func.get('signature', ''),
                    'template_params': func.get('template_params', ''),
                    'is_specialization': func.get('is_specialization', False),
                    'is_operator': func.get('is_operator', False),
                    'specifiers': func.get('specifiers', []),
                    'body': func.get('body', ''),
                    'calls': func.get('calls', []),
                    'used_identifiers': func.get('used_identifiers', []),
                    'line_start': func.get('line_start', 0),
                    'line_end': func.get('line_end', 0),
                    'comments': func.get('comments', [])
                })
            
            # Add to the dictionary
            file_function_dict[file_name] = function_details
        
        return file_function_dict
    
    except Exception as e:
        st.error(f"An error occurred while reading the JSON file: {e}")
        return {}

def search_function_by_name(results, target_name):
    """
    Search for a function by name across all analyzed files.
    
    Args:
        results (dict): The aggregated analysis results (as returned by `process_cpp_repository`).
        target_name (str): The name of the function to search for.
        
    Returns:
        list: A list of dictionaries containing function matches with their file paths and metadata.
    """
    matches = []
    for file_result in results.get('file_results', []):
        file_path = file_result.get('file_path', 'Unknown File')
        for func in file_result.get('functions', []):
            if func.get('normalized_name') == target_name:
                match = {
                    'file_path': file_path,
                    'name': func.get('name'),
                    'normalized_name': func.get('normalized_name'),
                    'signature': func.get('signature'),
                    'template_params': func.get('template_params', ''),
                    'is_specialization': func.get('is_specialization', False),
                    'is_operator': func.get('is_operator', False),
                    'specifiers': func.get('specifiers', []),
                    'line_start': func.get('line_start', 0),
                    'line_end': func.get('line_end', 0),
                    'comments': func.get('comments', []),
                    'body': func.get('body', '')
                }
                matches.append(match)
    return matches

def find_callers_of_function(results, target_function_name):
    """
    Find all functions that call a specific function by name across all analyzed files.

    Args:
        results (dict): The aggregated analysis results (as returned by `process_cpp_repository`).
        target_function_name (str): The name of the function to search for as a callee.

    Returns:
        list: A list of dictionaries with information about each calling function.
    """
    callers = []
    for file_result in results.get('file_results', []):
        file_path = file_result.get('file_path', 'Unknown File')
        for func in file_result.get('functions', []):
            if target_function_name in func.get('calls', []):
                callers.append({
                    'file_path': file_path,
                    'caller_name': func.get('name'),
                    'caller_signature': func.get('signature'),
                    'line_start': func.get('line_start'),
                    'line_end': func.get('line_end'),
                    'calls': func.get('calls'),
                    'comments': func.get('comments'),
                    'body': func.get('body')
                })
    return callers


