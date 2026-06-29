import argparse
import json
import os
import time
import uuid

import requests
from kubernetes import client, config

NAMESPACE = "kg-experiments"
VLLM_DEPLOYMENT = "vllm"
EMBEDDINGS_DEPLOYMENT = "embeddings"
SWEEP_LABEL = "sweep-id"


class MlflowClient:
    """Minimal MLflow REST client — avoids the heavy mlflow package."""

    def __init__(self):
        self.base_url = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
        self.auth = (
            os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
            os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
        )

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}/api/2.0/mlflow/{path}", json=body, auth=self.auth)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, body: dict) -> dict:
        r = requests.patch(f"{self.base_url}/api/2.0/mlflow/{path}", json=body, auth=self.auth)
        r.raise_for_status()
        return r.json()

    def get_or_create_experiment(self, name: str) -> str:
        r = requests.get(
            f"{self.base_url}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": name},
            auth=self.auth,
        )
        if r.status_code == 200:
            return r.json()["experiment"]["experiment_id"]
        body = self._post("experiments/create", {"name": name})
        return body["experiment_id"]

    def create_run(self, experiment_id: str, run_name: str) -> str:
        body = self._post("runs/create", {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "start_time": int(time.time() * 1000),
        })
        return body["run"]["info"]["run_id"]

    def log_params(self, run_id: str, params: dict):
        self._post("runs/log-batch", {
            "run_id": run_id,
            "params": [{"key": k, "value": str(v)} for k, v in params.items()],
        })

    def log_metrics(self, run_id: str, metrics: dict):
        ts = int(time.time() * 1000)
        self._post("runs/log-batch", {
            "run_id": run_id,
            "metrics": [{"key": k, "value": v, "timestamp": ts, "step": 0} for k, v in metrics.items()],
        })

    def end_run(self, run_id: str, status: str = "FINISHED"):
        self._post("runs/update", {
            "run_id": run_id,
            "status": status,
            "end_time": int(time.time() * 1000),
        })


def load_k8s_config():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def scale_deployment(apps_v1: client.AppsV1Api, name: str, replicas: int):
    apps_v1.patch_namespaced_deployment_scale(
        name=name,
        namespace=NAMESPACE,
        body={"spec": {"replicas": replicas}},
    )
    print(f"{name} scaled to {replicas} replicas")


def wait_for_deployment_ready(apps_v1: client.AppsV1Api, name: str, timeout: int = 300):
    print(f"Waiting for {name} to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=NAMESPACE)
        ready = dep.status.ready_replicas or 0
        desired = dep.spec.replicas or 0
        if desired > 0 and ready >= desired:
            print(f"{name} is ready ({ready}/{desired})")
            return
        time.sleep(10)
    raise TimeoutError(f"{name} not ready after {timeout}s")


def build_job_manifest(
    sweep_id: str,
    run_index: int,
    cfg: dict,
    parent_run_id: str,
    image: str,
    cpu_request: str,
    memory_request: str,
    stub: bool,
) -> client.V1Job:
    job_name = f"kg-builder-{sweep_id}-{run_index}"
    data_uri = os.environ.get("DATA_URI", "data/fnspid_sample_nasdaq_long_text.csv")
    if stub:
        container_args = None
        command = ["sh", "-c", f"echo 'stub job {run_index} cfg={cfg}'; sleep 5; echo done"]
    else:
        command = None
        container_args = [
            "--data", data_uri,
            "--steps", str(cfg.get("steps", 3)),
            "--candidates", str(cfg.get("candidates", 2)),
            "--feature-mode", cfg.get("feature_mode", "path"),
            "--time-window-days", str(cfg.get("time_window_days", 150)),
        ]
    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=NAMESPACE,
            labels={SWEEP_LABEL: sweep_id},
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=3600,
            backoff_limit=0,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    service_account_name="sweep-runner",
                    containers=[
                        client.V1Container(
                            name="kg-builder",
                            image=image,
                            command=command,
                            args=container_args,
                            env_from=[
                                client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="kg-config")),
                                client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="kg-secrets")),
                            ],
                            env=[
                                client.V1EnvVar(name="MLFLOW_PARENT_RUN_ID", value=parent_run_id),
                            ],
                            resources=client.V1ResourceRequirements(
                                requests={"cpu": cpu_request, "memory": memory_request},
                                limits={"cpu": cpu_request, "memory": memory_request},
                            ),
                        )
                    ],
                )
            ),
        ),
    )


def wait_for_job(batch_v1: client.BatchV1Api, job_name: str):
    while True:
        job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
        if job.status.completion_time:
            print(f"  {job_name} completed")
            return True
        if job.status.failed:
            print(f"  {job_name} failed")
            return False
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="JSON array of experiment configs")
    parser.add_argument("--n-workers", type=int, default=2, help="vLLM replicas during sweep")
    parser.add_argument("--n-embedding-workers", type=int, default=2, help="embedding replicas during sweep")
    parser.add_argument("--image", default=os.environ.get("KG_BUILDER_IMAGE", "kg-orchestrator:latest"))
    parser.add_argument("--cpu-request", default="1")
    parser.add_argument("--memory-request", default="2Gi")
    parser.add_argument("--stub", action="store_true", help="use busybox stub instead of real image")
    args = parser.parse_args()

    if args.stub:
        args.image = "busybox"

    configs = json.loads(args.configs)
    sweep_id = str(uuid.uuid4())[:8]

    mlflow = MlflowClient()
    tracking_enabled = bool(mlflow.base_url)

    load_k8s_config()
    apps_v1 = client.AppsV1Api()
    batch_v1 = client.BatchV1Api()

    parent_run_id = "local"
    if tracking_enabled:
        experiment_id = mlflow.get_or_create_experiment("kg-sweep")
        parent_run_id = mlflow.create_run(experiment_id, f"sweep-{sweep_id}")
        mlflow.log_params(parent_run_id, {
            "n_configs": len(configs),
            "n_workers": args.n_workers,
            "sweep_id": sweep_id,
        })

    print(f"Starting sweep {sweep_id} — {len(configs)} configs, sequential")
    print(f"MLflow parent run: {parent_run_id}")

    if args.n_workers > 0:
        scale_deployment(apps_v1, VLLM_DEPLOYMENT, args.n_workers)
        scale_deployment(apps_v1, EMBEDDINGS_DEPLOYMENT, args.n_embedding_workers)
        wait_for_deployment_ready(apps_v1, VLLM_DEPLOYMENT, timeout=600)
        wait_for_deployment_ready(apps_v1, EMBEDDINGS_DEPLOYMENT, timeout=300)

    job_names = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Submitting: {cfg}")
        job = build_job_manifest(
            sweep_id, i, cfg, parent_run_id,
            image=args.image,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            stub=args.stub,
        )
        batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
        job_names.append(job.metadata.name)

    print(f"\nAll {len(job_names)} jobs submitted — waiting for completion...")
    failed_count = sum(
        0 if wait_for_job(batch_v1, name) else 1
        for name in job_names
    )

    if args.n_workers > 0:
        scale_deployment(apps_v1, VLLM_DEPLOYMENT, 0)
        scale_deployment(apps_v1, EMBEDDINGS_DEPLOYMENT, 0)

    if tracking_enabled:
        mlflow.log_metrics(parent_run_id, {
            "succeeded": len(configs) - failed_count,
            "failed": failed_count,
        })
        mlflow.end_run(parent_run_id)

    print(f"\nSweep complete. {len(configs) - failed_count}/{len(configs)} succeeded.")


if __name__ == "__main__":
    main()
