import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import requests
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

NAMESPACE = "kg-experiments"
VLLM_DEPLOYMENT = "vllm"
EMBEDDINGS_DEPLOYMENT = "embeddings"
SWEEP_LABEL = "sweep-id"

# cfg key -> (CLI flag, kind). "value" flags render `--flag <value>` when the key
# is present in cfg; "flag" flags render bare `--flag` when cfg[key] is truthy.
# Extend this table (rather than build_job_manifest) when main.py grows new CLI
# args that sweep configs should be able to vary.
OPTIONAL_ARG_SPECS = [
    ("single_addition", "--single-addition", "flag"),
    ("keep_regressing_steps", "--keep-regressing-steps", "flag"),
    ("auc_drop_tolerance", "--auc-drop-tolerance", "value"),
    ("embedding_type", "--embedding-type", "value"),
    ("local_model", "--local-model", "value"),
    ("lookback_days", "--lookback-days", "value"),
    ("min_chain_hops", "--min-chain-hops", "value"),
    ("max_chain_hops", "--max-chain-hops", "value"),
    ("path_uniqueness", "--path-uniqueness", "value"),
    ("max_metapath_hops", "--max-metapath-hops", "value"),
    ("train_ratio", "--train-ratio", "value"),
    ("semaphore_limit", "--semaphore-limit", "value"),
]


def build_optional_args(cfg: dict) -> str:
    """Render OPTIONAL_ARG_SPECS entries present in cfg as a CLI arg string."""
    parts = []
    for key, flag, kind in OPTIONAL_ARG_SPECS:
        if key not in cfg:
            continue
        if kind == "flag":
            if cfg[key]:
                parts.append(f" {flag}")
        else:
            parts.append(f" {flag} {cfg[key]}")
    return "".join(parts)


class MlflowClient:
    """Minimal MLflow REST client — avoids the heavy mlflow package."""

    def __init__(self):
        self.base_url = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
        # DagsHub accepts token as password for HTTP basic auth
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD") or os.environ.get("MLFLOW_TRACKING_TOKEN", "")
        self.auth = (
            os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
            password,
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
    neo4j_mode: str = "sidecar",
) -> client.V1Job:
    job_name = f"kg-builder-{sweep_id}-{run_index}"

    if "day_start" in cfg and "day_end" in cfg:
        date_args = f" --day-start {cfg['day_start']} --day-end {cfg['day_end']}"
    else:
        date_args = f" --time-window-days {cfg.get('time_window_days', 30)}"
    articles_per_day_arg = (
        f" --articles-per-day {cfg['articles_per_day']}"
        if "articles_per_day" in cfg else ""
    )
    evolution_prompt_arg = (
        f" --evolution-prompt {cfg['evolution_prompt']}"
        if "evolution_prompt" in cfg else ""
    )
    main_cmd = (
        f"python -m kg_builder_llm.main"
        f" --steps {cfg.get('steps', 1)}"
        f" --candidates {cfg.get('candidates', 1)}"
        f" --feature-mode {cfg.get('feature_mode', 'path')}"
        f"{date_args}"
        f"{articles_per_day_arg}"
        f"{evolution_prompt_arg}"
        f"{build_optional_args(cfg)}"
    )

    use_sidecar = neo4j_mode == "sidecar" and not stub

    if stub:
        command = ["sh", "-c", f"echo 'stub job {run_index} cfg={cfg}'; sleep 5; echo done"]
        container_args = None
    elif use_sidecar:
        _neo4j_wait = (
            "python3 -c \""
            "import socket,time\n"
            "while True:\n"
            " try: socket.create_connection(('localhost',7687),2).close(); break\n"
            " except: time.sleep(2)\n"
            "\""
        )
        command = ["sh", "-c"]
        container_args = [f"{_neo4j_wait} && {main_cmd}; RC=$?; touch /done/complete; exit $RC"]
    else:
        command = ["sh", "-c"]
        container_args = [main_cmd]

    # kg-builder env — sidecar mode overrides NEO4J_* to point at localhost
    kg_builder_env = [
        client.V1EnvVar(name="MLFLOW_PARENT_RUN_ID", value=parent_run_id),
    ]
    if "ticker" in cfg:
        kg_builder_env.append(client.V1EnvVar(name="TARGET_TICKER", value=cfg["ticker"]))
    if use_sidecar:
        kg_builder_env += [
            client.V1EnvVar(name="NEO4J_URI",      value="bolt://localhost:7687"),
            client.V1EnvVar(name="NEO4J_USERNAME", value="neo4j"),
            client.V1EnvVar(name="NEO4J_PASSWORD", value="sweeppass"),
            client.V1EnvVar(name="NEO4J_DATABASE", value="neo4j"),
        ]

    kg_builder_volume_mounts = (
        [client.V1VolumeMount(name="done", mount_path="/done")] if use_sidecar else []
    )

    containers = []
    volumes = []

    if use_sidecar:
        containers.append(client.V1Container(
            name="neo4j",
            image="neo4j:5-community",
            command=["sh", "-c"],
            args=[
                "/startup/docker-entrypoint.sh neo4j & "
                "NEO4J_PID=$!; "
                "until [ -f /done/complete ]; do sleep 2; done; "
                "kill $NEO4J_PID 2>/dev/null || true; "
                "wait $NEO4J_PID 2>/dev/null || true; "
                "exit 0"
            ],
            env=[
                client.V1EnvVar(name="NEO4J_AUTH",                              value="neo4j/sweeppass"),
                # APOC was missing entirely (confirmed live: apoc.path.expandConfig ->
                # ProcedureNotFound), which silently zeroed the entire path-embedding
                # feature block for the whole sweep to date (relationship_chains.py's
                # extract_chains_batch catches the exception and returns {}, triggering
                # the zero-vector fallback on every call). Installing + unrestricting it here.
                client.V1EnvVar(name="NEO4J_PLUGINS",                           value='["apoc"]'),
                client.V1EnvVar(name="NEO4J_dbms_security_procedures_unrestricted", value="apoc.*"),
                client.V1EnvVar(name="NEO4J_dbms_security_procedures_allowlist",    value="apoc.*"),
                client.V1EnvVar(name="NEO4J_server_memory_heap_initial__size",   value="256m"),
                client.V1EnvVar(name="NEO4J_server_memory_heap_max__size",       value="1g"),
                client.V1EnvVar(name="NEO4J_server_memory_pagecache_size",       value="128m"),
                client.V1EnvVar(name="NEO4J_server_http_enabled",                value="false"),
                client.V1EnvVar(name="NEO4J_server_https_enabled",               value="false"),
            ],
            ports=[client.V1ContainerPort(container_port=7687, name="bolt")],
            resources=client.V1ResourceRequirements(
                # Bumped 512Mi/768Mi -> 1Gi/1.5Gi: at 768Mi, a job crashed with
                # ServiceUnavailable/Connection refused mid-run once the graph
                # grew past a certain size (200-day window, ~2x the 100-day
                # pilot's data) -- Neo4j almost certainly got OOM-killed. Pod
                # restartPolicy=Never means a crashed sidecar never recovers,
                # so every later query in that job fails permanently.
                requests={"cpu": "250m", "memory": "1Gi"},
                limits={"cpu": "500m", "memory": "1536Mi"},
            ),
            volume_mounts=[
                client.V1VolumeMount(name="neo4j-data", mount_path="/data"),
                client.V1VolumeMount(name="neo4j-logs", mount_path="/logs"),
                client.V1VolumeMount(name="done",       mount_path="/done"),
            ],
        ))
        volumes += [
            client.V1Volume(name="neo4j-data", empty_dir=client.V1EmptyDirVolumeSource()),
            client.V1Volume(name="neo4j-logs", empty_dir=client.V1EmptyDirVolumeSource()),
            client.V1Volume(name="done",       empty_dir=client.V1EmptyDirVolumeSource()),
        ]

    containers.append(client.V1Container(
        name="kg-builder",
        image=image,
        command=command,
        args=container_args,
        env_from=[
            client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name="kg-config")),
            client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="kg-secrets")),
        ],
        env=kg_builder_env,
        resources=client.V1ResourceRequirements(
            requests={"cpu": cpu_request, "memory": memory_request},
            limits={"cpu": cpu_request, "memory": memory_request},
        ),
        volume_mounts=kg_builder_volume_mounts,
    ))

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
                    tolerations=[
                        client.V1Toleration(
                            key="nvidia.com/gpu",
                            operator="Exists",
                            effect="NoSchedule",
                        ),
                    ],
                    containers=containers,
                    volumes=volumes or None,
                )
            ),
        ),
    )


CREDENTIAL_REFRESH_INTERVAL = 45 * 60  # seconds; role-chained STS sessions cap at 1h


def _get_assumed_role_arn() -> "str | None":
    """IAM role ARN backing the current assumed-role session, or None if
    running on long-term credentials (e.g. a local IAM user) — those aren't
    role-chained and don't need proactive refresh."""
    try:
        out = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Arn", "--output", "text"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "assumed-role/" not in out:
        return None
    account_id = out.split(":")[4]
    role_name = out.split("assumed-role/")[1].split("/")[0]
    return f"arn:aws:iam::{account_id}:role/{role_name}"


def refresh_aws_credentials(role_arn: str) -> None:
    """Re-assume role_arn for a fresh session. Role chaining (OIDC -> github
    role -> this role) caps sessions at 1h regardless of the role's
    MaxSessionDuration, so long sweeps must refresh proactively instead of
    requesting a longer duration up front."""
    out = subprocess.run(
        ["aws", "sts", "assume-role", "--role-arn", role_arn,
         "--role-session-name", "sweep-refresh", "--output", "json"],
        capture_output=True, text=True, check=True,
    ).stdout
    creds = json.loads(out)["Credentials"]
    os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]
    print(f"  ↻ Refreshed AWS credentials (expires {creds['Expiration']})")
    # New STS credentials can take a few seconds to propagate to the EKS auth
    # webhook; the very next k8s API call has been observed to 401 without this.
    time.sleep(5)


class CredentialRefresher:
    """Proactively re-assumes the current role during long polling loops so
    the EKS auth exec plugin (which re-reads AWS_* env vars on each token
    refresh) never hits an expired role-chained session."""

    def __init__(self):
        self.role_arn = _get_assumed_role_arn()
        # Anchor to the ACTUAL session creation time (set by the workflow right
        # after "Assume deployment role"), not to when this object happens to be
        # instantiated -- GPU node provisioning + job submission can take 20+ min
        # before sweep.py even reaches this point, and counting from "now" caused
        # the chained session's 1h hard cap to be exceeded before the first
        # refresh fired (observed failure at 65m53s into a run).
        assumed_at = os.environ.get("ROLE_ASSUMED_AT_EPOCH")
        self.last_refresh = float(assumed_at) if assumed_at else time.time()

    def maybe_refresh(self):
        if self.role_arn and time.time() - self.last_refresh > CREDENTIAL_REFRESH_INTERVAL:
            refresh_aws_credentials(self.role_arn)
            self.last_refresh = time.time()


def dump_failed_job_logs(core_v1: client.CoreV1Api, job_name: str) -> None:
    """Print container logs for a failed job's pod(s) before the job's TTL
    (1h) garbage-collects them -- otherwise a failure is undiagnosable
    (this happened once already: cause of a real failure was unrecoverable
    by the time anyone looked)."""
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=f"job-name={job_name}"
        )
    except ApiException as e:
        print(f"  could not list pods for {job_name} to dump logs: {e}")
        return
    for pod in pods.items:
        pod_name = pod.metadata.name
        container_names = [c.name for c in pod.spec.containers]
        for c_name in container_names:
            print(f"\n  ----- logs: pod={pod_name} container={c_name} (last 200 lines) -----")
            try:
                logs = core_v1.read_namespaced_pod_log(
                    name=pod_name, namespace=NAMESPACE, container=c_name,
                    tail_lines=200, timestamps=True,
                )
                print(logs)
            except ApiException as e:
                print(f"  (could not fetch logs for {pod_name}/{c_name}: {e})")
            print(f"  ----- end logs: {pod_name}/{c_name} -----")


# Log lines that reveal whether the APOC-dependent chain block and the remote
# embedding service actually produced data, versus silently falling back.
# extract_chains_batch catches a failed expandConfig and returns {}, and both
# dedup passes only warn -- so a job can report success with the entire
# path-feature block zeroed. That is precisely what went undetected across the
# pre-fix sweep, and a green workflow does not distinguish the two cases.
DIAGNOSTIC_PATTERNS = (
    "Batch query executed successfully",
    "Batch query failed",
    "BATCH EXTRACTION COMPLETE",
    "Total chains extracted",
    "Total chains embedded",
    "Articles with chains",
    "No chains extracted",
    "No chains embedded",
    "Chain aggregation complete",
    "Aggregated embedding shape",
    "deduplication failed",
    "ProcedureNotFound",
    "apoc",
    "APOC",
    # The subgraph and topology blocks can go all-zero just as silently as the
    # path block did -- _hashed_histogram returns zeros with no log line at
    # all -- so capture their coverage lines too, not just APOC's.
    "Built feature vector",
    "Built temporal subgraph features",
    "Computed topology features",
    "No topology features found",
    "No features to aggregate",
    "FEATURE BLOCK",
    "Feature block",
    "Feature matrix shape",
)


def dump_job_diagnostics(core_v1: client.CoreV1Api, job_name: str) -> None:
    """Print the slice of a completed job's logs that shows whether the APOC
    chain extraction and the embedding service did real work.

    Gated behind --dump-logs: a full chunk's logs are far too large to put in
    the workflow output, but for a smoke-test run these lines are the only
    way to tell real work from a silent zero-vector fallback."""
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=f"job-name={job_name}"
        )
    except ApiException as e:
        print(f"  could not list pods for {job_name} to dump diagnostics: {e}")
        return
    for pod in pods.items:
        pod_name = pod.metadata.name
        for c_name in [c.name for c in pod.spec.containers]:
            try:
                # _preload_content=False: with the default the client hands back
                # the repr of a bytes object ('b"line\\nline"') as ONE string, so
                # splitlines() sees a single line and the whole container log
                # gets printed as one multi-megabyte line -- which GitHub's log
                # API then drops silently. Read the raw stream and decode it.
                resp = core_v1.read_namespaced_pod_log(
                    name=pod_name, namespace=NAMESPACE, container=c_name,
                    _preload_content=False,
                )
                logs = resp.data.decode("utf-8", "replace")
            except ApiException as e:
                print(f"  (could not fetch logs for {pod_name}/{c_name}: {e})")
                continue
            lines = logs.splitlines()
            matched = [ln for ln in lines if any(p in ln for p in DIAGNOSTIC_PATTERNS)]
            print(
                f"\n  ----- diagnostics: {pod_name}/{c_name} "
                f"({len(matched)} matched of {len(lines)} lines) -----"
            )
            # Truncate each line too: a single runaway line (a dumped Cypher
            # query, a stack trace with embedded data) is enough to blow the
            # per-line limit and take the whole record with it.
            for ln in matched[:400]:
                print(f"    {ln[:500]}")
            print(f"  ----- last 40 lines: {pod_name}/{c_name} -----")
            for ln in lines[-40:]:
                print(f"    {ln[:500]}")
            print(f"  ----- end diagnostics: {pod_name}/{c_name} -----")


def wait_for_job(
    batch_v1: client.BatchV1Api,
    core_v1: client.CoreV1Api,
    job_name: str,
    refresher: CredentialRefresher,
):
    while True:
        refresher.maybe_refresh()
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
        except ApiException as e:
            # A freshly-refreshed STS session can briefly 401 against the EKS auth
            # webhook before it propagates -- retry rather than killing the sweep.
            if e.status == 401:
                print(f"  transient 401 polling {job_name}, retrying in 10s...")
                time.sleep(10)
                continue
            raise
        if job.status.completion_time:
            print(f"  {job_name} completed")
            return True
        if job.status.failed:
            print(f"  {job_name} failed")
            dump_failed_job_logs(core_v1, job_name)
            return False
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="JSON array of experiment configs")
    parser.add_argument("--n-workers", type=int, default=2, help="vLLM replicas during sweep")
    parser.add_argument("--n-embedding-workers", type=int, default=2, help="embedding replicas during sweep")
    parser.add_argument("--image", default=os.environ.get("KG_BUILDER_IMAGE", "kg-orchestrator:latest"))
    parser.add_argument("--cpu-request", default="500m")
    parser.add_argument("--memory-request", default="2Gi")
    parser.add_argument("--stub", action="store_true", help="use busybox stub instead of real image")
    parser.add_argument(
        "--neo4j-mode",
        choices=["sidecar", "external"],
        default="sidecar",
        help="sidecar: ephemeral neo4j per job (default); external: use NEO4J_* from kg-secrets (AuraDB)",
    )
    parser.add_argument(
        "--dump-logs",
        action="store_true",
        help="after each job completes, print the APOC/embedding diagnostic lines from its "
             "pod logs (smoke-test runs only -- a full chunk's output is far too large)",
    )
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
    core_v1 = client.CoreV1Api()

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
        # GPU node provisioning + image pull + model load from the shared EFS
        # cache has real variance -- 600s flaked once, 1200s flaked again on a
        # later run (no evidence of load-related cause, looks like plain AWS
        # capacity/scheduling variance). Bumped with more margin this time.
        wait_for_deployment_ready(apps_v1, VLLM_DEPLOYMENT, timeout=1800)
        wait_for_deployment_ready(apps_v1, EMBEDDINGS_DEPLOYMENT, timeout=900)

    # Submit all jobs up front, then wait for all in parallel.
    submitted = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Submitting: {cfg}")
        job = build_job_manifest(
            sweep_id, i, cfg, parent_run_id,
            image=args.image,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            stub=args.stub,
            neo4j_mode=args.neo4j_mode,
        )
        batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
        submitted.append(job)

    print(f"\nAll {len(submitted)} jobs submitted — waiting for completion...")
    refresher = CredentialRefresher()
    failed_count = 0
    for job in submitted:
        if not wait_for_job(batch_v1, core_v1, job.metadata.name, refresher):
            failed_count += 1
        elif args.dump_logs:
            # Only on success -- a failure already dumped its full tail above.
            dump_job_diagnostics(core_v1, job.metadata.name)

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

    if failed_count > 0:
        # Previously exited 0 regardless -- GHA showed green on a run that
        # silently dropped a config. Surface partial failure as a real failure.
        sys.exit(1)


if __name__ == "__main__":
    main()
