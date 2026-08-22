# Serve DeepSeek-R1 with vLLM on Ironwood TPU

In this guide, we show how to serve and benchmark [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) with vLLM on Ironwood (TPU v7x) using GKE.

DeepSeek-R1 is 671B parameter Mixture-of-Experts (MoE) models (37B active parameters per token) utilizing Multi-head Latent Attention (MLA). We deploy them on a single **tpu7x-standard-4t** node pool (4 TPU v7x chips, 8 TensorCores) using a **2x2x1** topology and **Tensor Parallelism (TP) size of 8**, using FP8 activation all-gather, FP4 weight requantization, and DP-attention sharding.

---

## Verified Models

| Model | Parameters | Active Parameters | Min TPUs (Chips) | Topology | Hugging Face |
| :---- | :---- | :---- | :---- | :---- | :---- |
| DeepSeek-R1 | 671B | 37B | 4× (tpu7x-standard-4t) | 2x2x1 | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |

---

## Prerequisites

1. A GKE cluster with TPU v7x (Ironwood) support (GKE version `1.34.0-gke.2201000` or later).
2. Google Cloud SDK (`gcloud`) and `kubectl` installed and configured.
3. A Hugging Face account and access token for downloading model weights and tokenizer files.

---

## Step 1: Define Parameters

Export the required environment variables for your GCP project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="deepseek-r1-tpu-pool"
```

---

## Step 2: Create the TPU Node Pool

Create a TPU v7x (Ironwood) node pool with the `tpu7x-standard-4t` machine type (4 chips / 8 TensorCores):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=tpu7x-standard-4t \
    --cluster=${CLUSTER_NAME}
```

---

## Step 3: Setup Kubernetes Configurations

1. Configure `kubectl` credentials:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID}
```

2. Create the `vllm-deepseek` namespace:

```shell
kubectl create namespace vllm-deepseek
```

3. Create the Hugging Face token secret:

```shell
export HF_TOKEN="<YOUR_HUGGING_FACE_TOKEN>"
kubectl create secret generic hf-secret \
    --namespace=vllm-deepseek \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the provided [deepseek_r1-server.yaml](./deepseek_r1-server.yaml) manifest:

```shell
kubectl apply -f deepseek_r1-server.yaml
```

Monitor pod status:

```shell
kubectl get pods -n vllm-deepseek -w
```

Stream server logs:

```shell
kubectl logs -n vllm-deepseek deployment/vllm-deepseek-r1 -c vllm-tpu -f
```

When startup is complete, you should see:

```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

---

## Step 5: Interact with the Model

1. Port-forward the vLLM service locally:

```shell
kubectl port-forward -n vllm-deepseek service/vllm-deepseek-r1-service 8000:8000
```

2. Send a test request using `curl`:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Think step by step."}
        ],
        "max_tokens": 256,
        "temperature": 0.6
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` benchmarking client to evaluate serving throughput and latency for a **1K Input / 1K Output (1k/1k)** workload.

### Benchmark Manifest

Save the following manifest as `deepseek_r1-benchmark.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: deepseek-r1-bench
  namespace: vllm-deepseek
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-tpu:nightly-20260626-c539adc-cc79815
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -s -f http://vllm-deepseek-r1-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /tmp/inferencex && \
      cd /tmp/inferencex && \
      git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
      python3 /tmp/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-deepseek-r1-service \
        --port=8000 \
        --model=deepseek-ai/DeepSeek-R1 \
        --tokenizer=deepseek-ai/DeepSeek-R1 \
        --dataset-name=random \
        --random-input-len=1024 \
        --random-output-len=1024 \
        --random-range-ratio=0.8 \
        --num-prompts=512 \
        --max-concurrency=128 \
        --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply the benchmark pod:

```shell
kubectl apply -f deepseek_r1-benchmark.yaml
```

View the benchmark output logs:

```shell
kubectl logs -n vllm-deepseek deepseek-r1-bench -f
```

### Benchmark Results

#### Example Benchmark Output

```
============ Serving Benchmark Result ============
Successful requests:                     504
Failed requests:                         8
Maximum request concurrency:             128
Benchmark duration (s):                  434.28
Total input tokens:                      516096
Total generated tokens:                  516096
Request throughput (req/s):              1.16
Output token throughput (tok/s):         1188.39
Peak output token throughput (tok/s):    19624.00
Peak concurrent requests:                170.00
Total token throughput (tok/s):          2376.78
---------------Time to First Token----------------
Mean TTFT (ms):                          14343.53
Median TTFT (ms):                        13775.80
P99 TTFT (ms):                           93084.55
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          80.50
Median TPOT (ms):                        74.69
P99 TPOT (ms):                           137.32
---------------Inter-token Latency----------------
Mean ITL (ms):                           80.50
Median ITL (ms):                         0.01
P99 ITL (ms):                            271.92
----------------End-to-end Latency----------------
Mean E2EL (ms):                          96699.08
Median E2EL (ms):                        81872.88
P99 E2EL (ms):                           207312.47
==================================================
```

#### Summary Metrics

| Workload (Input / Output) | Output Token Throughput (Total) | Output Token Throughput (Per Chip) | Mean TTFT | Mean TPOT | Mean ITL |
| :------------------------ | :------------------------------ | :--------------------------------- | :-------- | :-------- | :------- |
| 1k / 1k                   | 1188.39 tok/s                   | 297.10 tok/s/chip                  | 14.34 s   | 80.50 ms  | 80.50 ms |

**Note**: These benchmark results are measured using the vLLM benchmark client on a single-node Ironwood TPU v7x (topology `2x2x1`, 4 chips / 8 TensorCores) with FP8 KV cache, FP4 weight requantization, and DP-attention sharding. The development team is continuously optimizing performance; results are subject to change as newer software versions and compiler optimizations are released.

Clean up the benchmark pod when finished:

```shell
kubectl delete -f deepseek_r1-benchmark.yaml
```

---

## Step 7: Cleanup

When you are done testing, delete the server and the node pool:

1. Delete the Kubernetes deployment and resources:

```shell
kubectl delete -f deepseek_r1-server.yaml
```

2. (Optional) Delete the TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
