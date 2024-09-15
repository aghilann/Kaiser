# serialize_workflow.py

import ast
import json
import uuid
import io
import pickle

def serialize_data(data):
    # Serialize data using pickle
    data_bytes = pickle.dumps(data)
    data_stream = io.BytesIO(data_bytes)
    length = len(data_bytes)
    content_type = 'application/octet-stream'  # Standard content type for binary data
    return data_stream, length, content_type

def deserialize_data(data_bytes):
    # Deserialize data using pickle
    data = pickle.loads(data_bytes)
    return data

class WorkflowSerializer(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code
        self.tasks = {}
        self.workflow = {}
        self.current_workflow = None
        self.task_order = []
        self.imports = []
        self.execution_id = uuid.uuid4()

    def visit_Import(self, node):
        self.imports.append(ast.get_source_segment(self.source_code, node))

    def visit_ImportFrom(self, node):
        self.imports.append(ast.get_source_segment(self.source_code, node))

    def visit_FunctionDef(self, node):
        # Check if the function is decorated with @task or @workflow
        decorators = node.decorator_list
        is_task = False
        is_workflow = False
        container_image = None

        for decorator in decorators:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'task':
                    is_task = True
                    # Extract arguments from decorator
                    for kw in decorator.keywords:
                        if kw.arg == 'container_image':
                            container_image = kw.value.value  # For Python >= 3.8
            elif isinstance(decorator, ast.Name):
                if decorator.id == 'task':
                    is_task = True
                elif decorator.id == 'workflow':
                    is_workflow = True

        if is_task:
            # It's a task
            function_name = node.name
            function_code = ast.get_source_segment(self.source_code, node)
            args = [arg.arg for arg in node.args.args]
            # We will generate 'code_to_execute' later when we have outputs
            self.tasks[function_name] = {
                'name': function_name,
                'code': function_code,
                'inputs': args,
                'outputs': [],  # Will be filled later
                'container_image': container_image
            }
        elif is_workflow:
            # It's a workflow
            self.current_workflow = node.name
            # Parse the workflow body
            self.parse_workflow_body(node.body)
        self.generic_visit(node)

    def parse_workflow_body(self, body):
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                # e.g., X, y = load_data()
                targets = self.extract_targets(stmt.targets)
                if isinstance(stmt.value, ast.Call):
                    func_name = self.get_func_name(stmt.value.func)
                    if func_name in self.tasks:
                        self.task_order.append({
                            'task': func_name,
                            'outputs': targets,
                            'inputs': self.get_call_args(stmt.value)
                        })
            elif isinstance(stmt, ast.Expr):
                # e.g., preprocess_data(X, y)
                if isinstance(stmt.value, ast.Call):
                    func_name = self.get_func_name(stmt.value.func)
                    if func_name in self.tasks:
                        self.task_order.append({
                            'task': func_name,
                            'outputs': [],
                            'inputs': self.get_call_args(stmt.value)
                        })

    def extract_targets(self, targets):
        extracted = []
        for t in targets:
            if isinstance(t, ast.Name):
                extracted.append(t.id)
            elif isinstance(t, ast.Tuple):
                extracted.extend(self.extract_targets(t.elts))
        return extracted

    def get_func_name(self, func):
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        else:
            return None

    def get_call_args(self, call_node):
        args = []
        for arg in call_node.args:
            if isinstance(arg, ast.Name):
                args.append((arg.id, False))
            elif isinstance(arg, ast.Constant):
                args.append((arg.value, True))
            else:
                arg_value = ast.unparse(arg) if hasattr(ast, 'unparse') else ast.dump(arg)
                args.append((arg_value, True))
        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Name):
                value = kw.value.id
                is_literal = False
            elif isinstance(kw.value, ast.Constant):
                value = kw.value.value
                is_literal = True
            else:
                value = ast.unparse(kw.value) if hasattr(ast, 'unparse') else ast.dump(kw.value)
                is_literal = True
            args.append((f"{kw.arg}={value}", is_literal))
        return args

    def generate_code_to_execute(self, function_name, args, outputs):
        serialization_functions = """
def serialize_data(data):
    import io
    import pickle
    data_bytes = pickle.dumps(data)
    data_stream = io.BytesIO(data_bytes)
    length = len(data_bytes)
    content_type = 'application/octet-stream'
    return data_stream, length, content_type

def deserialize_data(data_bytes):
    import pickle
    data = pickle.loads(data_bytes)
    return data
        """

        # Generate code to retrieve arguments
        arg_loading_code = []
        args_list = []
        for arg_name, is_literal in args:
            if is_literal:
                # Include the literal value directly in the function call
                args_list.append(repr(arg_name))
            else:
                # Retrieve from MinIO
                code_line = f"""
# Retrieve argument: {arg_name}
response = minio_client.get_object('{self.execution_id}', '{arg_name}')
data_bytes = response.read()
{arg_name} = deserialize_data(data_bytes)
print(f"Loaded argument '{arg_name}'")
                """
                arg_loading_code.append(code_line.strip())
                args_list.append(arg_name)

        args_str = ', '.join(args_list)
        function_call = f"result = {function_name}({args_str})"

        # Generate code to store the result(s)
        result_storage_lines = []
        if outputs:
            if len(outputs) == 1:
                # Single output
                output_var = outputs[0]
                result_storage_lines.append(f"{output_var} = result")
                result_storage_lines.append(f"""
# Serialize and store the result
data_stream, length, content_type = serialize_data({output_var})
minio_client.put_object('{self.execution_id}', '{output_var}', data_stream, length, content_type=content_type)
print({output_var})
                """.strip())
            else:
                # Multiple outputs
                outputs_str = ', '.join(outputs)
                result_storage_lines.append(f"{outputs_str} = result")
                for output_var in outputs:
                    result_storage_lines.append(f"""
# Serialize and store '{output_var}'
data_stream, length, content_type = serialize_data({output_var})
minio_client.put_object('{self.execution_id}', '{output_var}', data_stream, length, content_type=content_type)
print({output_var})
                    """.strip())

        # Combine all code components
        code_lines = [
            "# Code to execute",
            "from minio import Minio",
            "import io",
            "import pickle",
            "# Initialize MinIO client",
            "minio_client = Minio('minio-service:9000', access_key='minio', secret_key='minio123', secure=False)",
            "# Create bucket if it does not exist",
            f"execution_id = '{self.execution_id}'",
            "buckets = [bucket.name for bucket in minio_client.list_buckets()]",
            "if execution_id not in buckets: minio_client.make_bucket(execution_id)",
            "# Serialization functions",
            serialization_functions.strip(),
            "# Retrieve arguments",
            '\n'.join(arg_loading_code),
            "# Call the function",
            function_call,
            "# Store the result(s)",
            '\n'.join(result_storage_lines)
        ]
        return '\n'.join(code_lines)

    def main(self):
        # After parsing, generate 'code_to_execute' for each task in the task order
        for task in self.task_order:
            function_name = task['task']
            inputs = task['inputs']
            outputs = task['outputs']
            # Update task outputs in self.tasks
            self.tasks[function_name]['outputs'] = outputs
            code_to_execute = self.generate_code_to_execute(function_name, inputs, outputs)
            self.tasks[function_name]['code_to_execute'] = code_to_execute

    def serialize_workflow(self):
        # Build the JSON representation
        workflow_json = {
            'workflow': {
                'name': self.current_workflow,
                'tasks': self.task_order,
            },
            'tasks': self.tasks,
            'imports': self.imports,
            'execution_id': str(self.execution_id),
        }
        return workflow_json

def main():
    # Read the source code from the file
    with open('mnist.py', 'r') as f:
        source_code = f.read()

    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Create an instance of the serializer and visit the AST nodes
    serializer = WorkflowSerializer(source_code)
    serializer.visit(tree)
    serializer.main()  # Generate code_to_execute for tasks

    workflow_json = serializer.serialize_workflow()

    # Output the JSON to a file or print it
    with open('workflow.json', 'w') as f:
        json.dump(workflow_json, f, indent=4)

    print("Workflow serialized to 'workflow.json' successfully.")

if __name__ == "__main__":
    main()
