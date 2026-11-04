#!/usr/bin/env bash
# Runs one model family (two AG News pools) sequentially on one GPU.
set -euo pipefail
FAMILY="${1:?usage: bash run_one_family.sh qwen|llama|mistral|phi}"
case "$FAMILY" in qwen|llama|mistral|phi) ;; *) echo 'Unknown family' >&2; exit 2;; esac
ROOT="${E7_REPO_ROOT:-$PWD}"
E7_HOST_PORT="${E7_HOST_PORT:-18080}"
if ! [[ "$E7_HOST_PORT" =~ ^[0-9]+$ ]] || (( 10#$E7_HOST_PORT < 1024 || 10#$E7_HOST_PORT > 65535 )); then
  echo "Invalid E7_HOST_PORT: $E7_HOST_PORT (use 1024..65535)" >&2; exit 2
fi
TASKROOT="$ROOT/AuditRepair-v8.4/E7_tasks"
CODE="$ROOT/AuditRepair-v8.4/code"
OUT="${E7_OUTPUT_ROOT:-$ROOT/e7_collected}"
HOST_HF_CACHE="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
IMAGE="${VLLM_IMAGE:?Set VLLM_IMAGE to an immutable vLLM image digest, e.g. vllm/vllm-openai@sha256:...}"
[[ "$IMAGE" == *@sha256:* ]] || { echo 'Refusing unpinned vLLM image; choose a digest' >&2; exit 2; }
command -v docker >/dev/null && command -v python3 >/dev/null || { echo 'Docker and python3 are required' >&2; exit 2; }
[[ -f "$TASKROOT/E7_GOLD_BLIND_MANIFEST.json" ]] || { echo "Missing worklists at $TASKROOT" >&2; exit 2; }
mkdir -p "$OUT/responses" "$OUT/provenance" "$HOST_HF_CACHE"
# Fail before launching if another listener already owns the requested host port.
if python3 - "$E7_HOST_PORT" <<'PYPORT'
import socket,sys
port=int(sys.argv[1]);s=socket.socket();s.settimeout(1)
try:
    busy = s.connect_ex(('127.0.0.1',port)) == 0
finally:
    s.close()
sys.exit(0 if busy else 1)
PYPORT
then
  echo "Host port $E7_HOST_PORT is already serving connections; choose another with E7_HOST_PORT=..." >&2
  exit 2
fi
python3 "$(dirname "$0")/preflight_tasks.py" --tasks "$TASKROOT" --family "$FAMILY"
readarray -t DETAILS < <(python3 - "$TASKROOT" "$FAMILY" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]+'/E7_GOLD_BLIND_MANIFEST.json'))
m=next(y for y in x['models'] if y['family']==sys.argv[2]);print(m['model_id']);print(m['revision'])
PY
)
MODEL="${DETAILS[0]}"; REVISION="${DETAILS[1]}"
NAME="auditrepair-e7-${FAMILY}"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "Existing container $NAME found. Refusing to replace. Inspect or remove explicitly: docker rm -f $NAME" >&2; exit 2
fi
# Evidence must not contain authentication secrets. Captures pinned image/revision and actual container ID.
python3 - "$OUT/provenance/${FAMILY}_start.json" "$FAMILY" "$MODEL" "$REVISION" "$IMAGE" <<'PY'
import json,sys,datetime
json.dump(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),family=sys.argv[2],model=sys.argv[3],revision=sys.argv[4],image=sys.argv[5],collection_status='starting'),open(sys.argv[1],'w'),indent=2)
PY
DOCKER_ARGS=(run -d --name "$NAME" --gpus all --ipc=host -p "127.0.0.1:${E7_HOST_PORT}:8000" -v "$HOST_HF_CACHE:/root/.cache/huggingface" )
# Token only passed as environment; never printed. Hugging Face gated models need HF_TOKEN.
if [[ -n "${HF_TOKEN:-}" ]]; then DOCKER_ARGS+=(--env HF_TOKEN); fi
DOCKER_ARGS+=("$IMAGE" --model "$MODEL" --revision "$REVISION" --tokenizer-revision "$REVISION" --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-seqs 8)
CID=$(docker "${DOCKER_ARGS[@]}")
trap 'echo "Server retained as '"$NAME"' for inspection; remove with docker rm -f '"$NAME"'"' EXIT
# record runtime facts (no credentials)
docker inspect "$CID" --format '{{json .Image}}' > "$OUT/provenance/${FAMILY}_container_image_id.json"
docker inspect "$CID" --format '{{json .Config.Cmd}}' > "$OUT/provenance/${FAMILY}_server_command.json"
echo "Started $NAME on host port $E7_HOST_PORT; loading $MODEL at $REVISION"
ready=0
for i in $(seq 1 360); do
 if python3 - "$MODEL" "$E7_HOST_PORT" <<'PY'
import json,sys,urllib.request
try:
 x=json.load(urllib.request.urlopen('http://127.0.0.1:'+sys.argv[2]+'/v1/models',timeout=3))
 assert sys.argv[1] in [m['id'] for m in x['data']]
except Exception:sys.exit(1)
PY
 then ready=1;break; fi
 if ! docker ps --format '{{.Names}}' | grep -Fxq "$NAME"; then docker logs --tail 60 "$NAME"; exit 1;fi
 sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Server readiness timeout. Inspect docker logs' >&2; exit 1; }
for PANEL in ag_news_5000 ag_news_holdout; do
 python3 "$CODE/e7_collect_one_model.py" \
   --tasks "$TASKROOT/${PANEL}_${FAMILY}_tasks.jsonl" \
   --output "$OUT/responses/${PANEL}_${FAMILY}_responses.jsonl" \
   --base-url "http://127.0.0.1:${E7_HOST_PORT}/v1" --timeout 180
done
python3 "$(dirname "$0")/preflight_tasks.py" --tasks "$TASKROOT" --family "$FAMILY" --responses "$OUT/responses"
echo "Family $FAMILY complete. Check $OUT/responses and $OUT/provenance."
echo "When satisfied stop the server explicitly: docker rm -f $NAME"
