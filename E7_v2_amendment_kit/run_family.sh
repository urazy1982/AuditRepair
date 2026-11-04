#!/usr/bin/env bash
# E7-RC2: full-family amended collection. Never overwrites original E7 files.
set -euo pipefail
FAMILY="${1:?Usage: bash E7_v2_amendment_kit/run_family.sh mistral|phi}"
case "$FAMILY" in mistral|phi) ;; *) echo 'Only mistral or phi can be recollected in RC2' >&2; exit 2 ;; esac
ROOT="${E7_REPO_ROOT:-$PWD}"
PORT="${E7_HOST_PORT:-18080}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((10#$PORT >= 1024 && 10#$PORT <= 65535)) || { echo 'Invalid E7_HOST_PORT' >&2; exit 2; }
if [[ -z "${VLLM_IMAGE:-}" ]]; then
  # Reuse *recorded* original digest, never a floating latest tag.
  VLLM_IMAGE="$(python3 - "$ROOT" "$FAMILY" <<'PYIMAGE'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])/'e7_collected/provenance'/f'{sys.argv[2]}_start.json'
if not p.exists():sys.exit('Missing original immutable image evidence: '+str(p))
print(json.loads(p.read_text())['image'])
PYIMAGE
)"
  export VLLM_IMAGE
fi
[[ "$VLLM_IMAGE" == *@sha256:* ]] || { echo 'VLLM_IMAGE is not an immutable digest' >&2; exit 2; }

TASKROOT="$ROOT/e7_rc2/tasks"
OUT="$ROOT/e7_collected_rc2"
MANIFEST="$ROOT/e7_rc2/E7_RC2_MANIFEST.json"
KIT="$ROOT/E7_v2_amendment_kit"
[[ -f "$MANIFEST" && -f "$TASKROOT/ag_news_5000_${FAMILY}_tasks.jsonl" ]] || { echo 'Run prepare.py first' >&2; exit 2; }
command -v docker >/dev/null; command -v python3 >/dev/null
mkdir -p "$OUT/responses" "$OUT/provenance" "${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
# fail closed on occupied host port; don't stop unrelated servers
if python3 - "$PORT" <<'PY'
import socket,sys
s=socket.socket();s.settimeout(1)
try:occupied=s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0
finally:s.close()
sys.exit(0 if occupied else 1)
PY
then echo "Host port $PORT is occupied. Do not stop unrelated containers." >&2; exit 2; fi
readarray -t DETAILS < <(python3 - "$MANIFEST" "$FAMILY" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));d=next(x for x in m['models'] if x['family']==sys.argv[2]);print(d['model_id']);print(d['revision'])
PY
)
MODEL="${DETAILS[0]}"; REV="${DETAILS[1]}"
NAME="auditrepair-e7rc2-${FAMILY}"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "Container $NAME exists; inspect and remove explicitly before rerun" >&2;exit 2
fi
HFROOT="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
ARGS=(run -d --name "$NAME" --gpus all --ipc=host -p "127.0.0.1:${PORT}:8000" -v "$HFROOT:/root/.cache/huggingface")
[[ -z "${HF_TOKEN:-}" ]] || ARGS+=(--env HF_TOKEN)
ARGS+=("$VLLM_IMAGE" --model "$MODEL" --revision "$REV" --tokenizer-revision "$REV" --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-seqs 8)
CID=$(docker "${ARGS[@]}")
python3 - "$OUT/provenance/${FAMILY}_amendment_start.json" "$MODEL" "$REV" "$VLLM_IMAGE" <<'PY'
import json,sys,datetime
json.dump({'amendment':'E7-RC2','model':sys.argv[2],'revision':sys.argv[3],'image':sys.argv[4],
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
docker inspect "$CID" --format '{{json .Image}}' > "$OUT/provenance/${FAMILY}_container_image_id.json"
docker inspect "$CID" --format '{{json .Config.Cmd}}' > "$OUT/provenance/${FAMILY}_server_command.json"
echo "Started $NAME ($MODEL at $REV) on port $PORT"
ready=0
for _ in $(seq 1 360); do
  if python3 - "$MODEL" "$PORT" <<'PY'
import json,sys,urllib.request
try:
 r=json.load(urllib.request.urlopen('http://127.0.0.1:'+sys.argv[2]+'/v1/models',timeout=3))
 assert sys.argv[1] in [v['id'] for v in r['data']]
except Exception:sys.exit(1)
PY
  then ready=1;break;fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$NAME";then docker logs --tail 50 "$NAME";exit 1;fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo 'vLLM readiness timeout' >&2;exit 1; }
for PANEL in ag_news_5000 ag_news_holdout;do
  python3 "$KIT/collect.py" --manifest "$MANIFEST" --tasks "$TASKROOT/${PANEL}_${FAMILY}_tasks.jsonl" \
    --output "$OUT/responses/${PANEL}_${FAMILY}_responses.jsonl" --base-url "http://127.0.0.1:${PORT}/v1"
done
echo "Family $FAMILY completed; source rows saved under $OUT."
echo "You may now stop only this server: docker rm -f $NAME"
