# kubectl Reference — kg-experiments

## Cluster

```bash
minikube start --driver=podman
minikube stop
kubectl config get-contexts
kubectl config use-context minikube
kubectl config use-context kg-experiments-eks
```

## Deploy

```bash
kubectl apply -k k8s/overlays/local     # local
kubectl apply -k k8s/overlays/aws       # production
kubectl delete -k k8s/overlays/local    # tear down
```

## Secrets (never committed to git)

```bash
kubectl create secret generic kg-secrets --from-env-file=.env -n kg-experiments
```

## Inspect

```bash
kubectl get all -n kg-experiments
kubectl get pods -n kg-experiments -w
kubectl describe pod <name> -n kg-experiments
kubectl logs <pod> -n kg-experiments --follow
kubectl events -n kg-experiments
```

## Debug

```bash
kubectl exec -it <pod> -n kg-experiments -- bash
kubectl port-forward svc/vllm-svc 8000:8000 -n kg-experiments
kubectl run curl-test --image=curlimages/curl -it --rm -n kg-experiments -- sh
```

## Jobs

```bash
kubectl get jobs -n kg-experiments
kubectl logs -n kg-experiments -l job-name=<name> --follow
kubectl delete job <name> -n kg-experiments
kubectl delete jobs --all -n kg-experiments
```

## Scale

```bash
kubectl scale deployment vllm --replicas=1 -n kg-experiments
kubectl scale deployment vllm --replicas=0 -n kg-experiments
```

## Helm

```bash
helm upgrade --install ollama ollama-helm/ollama -f k8s/helm-values/ollama-local.yaml -n kg-experiments --atomic
helm list -n kg-experiments
helm history ollama -n kg-experiments
helm uninstall ollama -n kg-experiments
```

## Storage

```bash
kubectl get pvc -n kg-experiments
kubectl describe pvc model-cache -n kg-experiments
kubectl get storageclass
```

## RBAC

```bash
kubectl auth can-i create jobs \
  --as=system:serviceaccount:kg-experiments:sweep-runner \
  -n kg-experiments
```
