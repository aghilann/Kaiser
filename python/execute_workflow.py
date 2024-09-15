import json
import uuid
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from minio import Minio

# Load Kubernetes configuration
config.load_kube_config()

# Create API clients
batch_v1 = client.BatchV1Api()
core_v1 = client.CoreV1Api()

# Namespace where MinIO is running
NAMESPACE = 'data-orchestrator'

# MinIO service details
MINIO_SERVICE = 'minio-service'
MINIO_PORT = 9000
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET_KEY = 'minio123'

# Docker image with Python
DOCKER_IMAGE = 'python'  # Replace with a custom image if available

def ensure_namespace_exists(namespace):
    try:
        core_v1.read_namespace(name=namespace)
        print(f"Namespace '{namespace}' already exists.")
    except ApiException as e:
        if e.status == 404:
            # Namespace does not exist, so create it
            print(f"Namespace '{namespace}' not found. Creating it.")
            ns_metadata = client.V1ObjectMeta(name=namespace)
            ns_spec = client.V1Namespace(metadata=ns_metadata)
            core_v1.create_namespace(body=ns_spec)
        else:
            print(f"Exception when checking/creating namespace: {e}")
            raise

def sanitize_name(name):
    # Replace underscores with hyphens and ensure compliance with RFC 1123
    return name.lower().replace('_', '-')

def create_job(task_name, task_spec, execution_id):
    job_name = f"{sanitize_name(task_name)}-{str(uuid.uuid4())[:5]}"
    container_name = sanitize_name(task_name)
    metadata = client.V1ObjectMeta(name=job_name, namespace=NAMESPACE)

    # Prepare code without escaping quotes
    task_code = task_spec['code']
    code_to_execute = task_spec['code_to_execute']
    code_to_execute = code_to_execute.replace('{execution_id}', execution_id)
    code_to_execute = code_to_execute.replace('minio-service:9000', f"{MINIO_SERVICE}:{MINIO_PORT}")
    code_to_execute = code_to_execute.replace("access_key='minio'", f"access_key='{MINIO_ACCESS_KEY}'")
    code_to_execute = code_to_execute.replace("secret_key='minio123'", f"secret_key='{MINIO_SECRET_KEY}'")

    # Command to run in the container using a heredoc
    command = [
        'sh', '-c', f"""
pip install --no-cache-dir scikit-learn pandas numpy joblib minio && \
mkdir -p /app && \
cat << 'EOF' > /app/task.py
from typing import List, Tuple
{task_code}
{code_to_execute}
EOF
python /app/task.py
"""
    ]

    # Define environment variables
    env_vars = [
        client.V1EnvVar(name='EXECUTION_ID', value=execution_id),
    ]

    # Container spec
    container = client.V1Container(
        name=container_name,
        image=DOCKER_IMAGE,
        command=command,
        env=env_vars,
        image_pull_policy='IfNotPresent',
    )

    # Pod template
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={'job-name': job_name}),
        spec=client.V1PodSpec(
            restart_policy='Never',
            containers=[container]
        )
    )

    # Job spec with backoffLimit
    job_spec = client.V1JobSpec(template=template, backoff_limit=0, ttl_seconds_after_finished=60)

    # Job
    job = client.V1Job(api_version='batch/v1', kind='Job', metadata=metadata, spec=job_spec)

    # Create the Job
    try:
        batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
        print(f"Job '{job_name}' for task '{task_name}' created.")
    except ApiException as e:
        print(f"Exception when creating job: {e}")

    return job_name

def get_pod_name(job_name):
    # Get the Pod name associated with a specific Job
    try:
        pods = core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=f'job-name={job_name}')
        if pods.items:
            return pods.items[0].metadata.name
    except ApiException as e:
        print(f"Exception when listing pods for job {job_name}: {e}")
    return None

def fetch_pod_logs(pod_name):
    # Fetch and print logs for a given Pod
    try:
        logs = core_v1.read_namespaced_pod_log(name=pod_name, namespace=NAMESPACE, previous=False)
        print(f"Logs for Pod '{pod_name}':\n{logs}")
        return logs
    except ApiException as e:
        print(f"Exception when reading logs for pod {pod_name}: {e}")
        return str(e)

def log_failure_details(job_name, pod_name, failure_reason, logs):
    # Write the failure details to a log file
    with open('container_failures.log', 'a') as log_file:
        log_file.write(f"Job: {job_name}, Pod: {pod_name}\n")
        log_file.write(f"Failure Reason: {failure_reason}\n")
        log_file.write(f"Logs:\n{logs}\n")
        log_file.write("="*80 + "\n")

def wait_for_job(job_name):
    pod_name = get_pod_name(job_name)
    while True:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
            if job.status.succeeded == 1:
                print(f"Job '{job_name}' completed successfully.")
                logs = fetch_pod_logs(pod_name)
                log_failure_details(job_name, pod_name, "Success", logs)
                break
            elif job.status.failed is not None and job.status.failed > 0:
                print(f"Job '{job_name}' failed.")
                if pod_name:
                    logs = fetch_pod_logs(pod_name)
                    failure_reason = "Job failed"
                    log_failure_details(job_name, pod_name, failure_reason, logs)
                break
            else:
                if pod_name:
                    # Check if the pod is in a failed state
                    pod = core_v1.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
                    if pod.status.phase == 'Failed':
                        print(f"Pod '{pod_name}' failed.")
                        logs = fetch_pod_logs(pod_name)
                        failure_reason = "Pod failed"
                        log_failure_details(job_name, pod_name, failure_reason, logs)
                        break
                else:
                    print(f"Pod not found for job '{job_name}'.")
                    break
        except ApiException as e:
            print(f"Exception when reading job or pod status: {e}")
            break
        time.sleep(1)

def topological_sort(workflow):
    # Build outputs map: variable name -> task that produces it
    outputs_map = {}
    for task_info in workflow['tasks']:
        task_name = task_info['task']
        outputs = task_info.get('outputs', [])
        for output in outputs:
            outputs_map[output] = task_name

    # Build the dependencies graph
    dependencies = {task_info['task']: set() for task_info in workflow['tasks']}

    for task_info in workflow['tasks']:
        task_name = task_info['task']
        inputs = [arg[0] for arg in task_info.get('inputs', []) if not arg[1]]  # Exclude literals
        for input_var in inputs:
            if input_var in outputs_map:
                dep_task = outputs_map[input_var]
                if dep_task != task_name:
                    dependencies[task_name].add(dep_task)

    # Perform topological sort using Kahn's algorithm
    S = [task for task in dependencies if len(dependencies[task]) == 0]
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in dependencies:
            if n in dependencies[m]:
                dependencies[m].remove(n)
                if len(dependencies[m]) == 0 and m not in L and m not in S:
                    S.append(m)

    if any(len(dependencies[task]) > 0 for task in dependencies):
        raise Exception("Cyclic dependency detected")

    return ["load_data", "preprocess_data", "split_data", "train_model", "evaluate_model"]  # Hardcoded for now,
    return L

def main():
    # Ensure the namespace exists
    ensure_namespace_exists(NAMESPACE)

    # Read the serialized workflow JSON
    with open('workflow.json', 'r') as f:
        workflow_json = json.load(f)

    workflow = workflow_json['workflow']
    tasks = workflow_json['tasks']
    execution_id = workflow_json['execution_id']

    print(f"Starting execution of workflow '{workflow['name']}' with execution ID '{execution_id}'.")

    # Compute topological order
    task_order = topological_sort(workflow)

    print("Task execution order:", task_order)

    # Execute tasks in topological order
    for task_name in task_order:
        task_spec = tasks[task_name]

        # Create the Kubernetes Job
        job_name = create_job(task_name, task_spec, execution_id)

        time.sleep(10)

        # Wait for the Job to complete
        wait_for_job(job_name)

    print(f"Workflow '{workflow['name']}' execution completed.")

if __name__ == '__main__':
    main()
