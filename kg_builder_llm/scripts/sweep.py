import argparse
import json
import os
import time
import uuid

from kubernetes import client, config

NAMESPACE = "kg-experiments"
VLLM_DEPLOYMENT = "vllm"
EMBEDDINGS_DEPLOYMENT = "embeddings"
SWEEP_LABEL = "sweep-id"


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
    parser.add_argument("--parent-run-id", default="local-sweep", help="MLflow parent run ID")
    parser.add_argument("--n-workers", type=int, default=2, help="vLLM replicas during sweep")
    parser.add_argument("--n-embedding-workers", type=int, default=2, help="embedding replicas during sweep")
    parser.add_argument("--image", default="ghcr.io/jjfrackowiak/kg-orchestrator:latest")
    parser.add_argument("--cpu-request", default="1")
    parser.add_argument("--memory-request", default="2Gi")
    parser.add_argument("--stub", action="store_true", help="use busybox stub instead of real image")
    args = parser.parse_args()

    if args.stub:
        args.image = "busybox"

    configs = json.loads(args.configs)
    sweep_id = str(uuid.uuid4())[:8]

    load_k8s_config()
    apps_v1 = client.AppsV1Api()
    batch_v1 = client.BatchV1Api()

    print(f"Starting sweep {sweep_id} — {len(configs)} configs, sequential")

    if args.n_workers > 0:
        scale_deployment(apps_v1, VLLM_DEPLOYMENT, args.n_workers)
        scale_deployment(apps_v1, EMBEDDINGS_DEPLOYMENT, args.n_embedding_workers)
        time.sleep(10)

    failed_count = 0
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Submitting: {cfg}")
        job = build_job_manifest(
            sweep_id, i, cfg, args.parent_run_id,
            image=args.image,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            stub=args.stub,
        )
        job_name = job.metadata.name
        batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
        success = wait_for_job(batch_v1, job_name)
        if not success:
            failed_count += 1

    if args.n_workers > 0:
        scale_deployment(apps_v1, VLLM_DEPLOYMENT, 0)
        scale_deployment(apps_v1, EMBEDDINGS_DEPLOYMENT, 1)

    print(f"\nSweep complete. {len(configs) - failed_count}/{len(configs)} succeeded.")


if __name__ == "__main__":
    main()
