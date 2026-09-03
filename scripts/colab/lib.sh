# Shared helpers for laptop Colab scripts. Source from other scripts:
#   source "$(dirname "$0")/lib.sh"

colab_session_active() {
  local cli="${1:-colab}"
  local session="${2:-video-retrieval}"
  local output=""

  if ! output=$("$cli" status -s "$session" 2>&1); then
    return 1
  fi
  local lower
  lower=$(printf '%s' "$output" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    *"no active session"*|*"not found"*) return 1 ;;
  esac
  return 0
}

# Check a directory exists on the Colab VM without sys.exit(0).
# IPython prints "An exception has occurred" for SystemExit(0) even on success.
colab_remote_dir_exists() {
  local cli="$1"
  local session="$2"
  local remote_dir="$3"
  local output=""
  local snippet

  snippet=$(python3 - "$remote_dir" <<'PY'
import json, sys
path = sys.argv[1]
print(
    "from pathlib import Path\n"
    f"p = Path({json.dumps(path)})\n"
    "print('COLAB_PATH_OK' if p.is_dir() else 'COLAB_PATH_MISSING')\n"
)
PY
)
  output=$(printf '%s\n' "$snippet" | "$cli" exec -s "$session" --timeout 30 2>&1) || true
  printf '%s\n' "$output"
  printf '%s\n' "$output" | grep -q "COLAB_PATH_OK"
}

# Mount Google Drive on the Colab VM via CLI (works without notebook UI).
# drive.mount() inside raw python/console cannot auth — use this instead.
colab_ensure_drive_mounted() {
  local cli="${1:-colab}"
  local session="${2:-video-retrieval}"
  local mount_path="${3:-/content/drive}"

  echo "Mounting Google Drive at $mount_path (colab drivemount)..."
  if ! "$cli" drivemount -s "$session" "$mount_path"; then
    echo "Drive mount failed. Try again, or open the notebook UI:" >&2
    echo "  colab url -s $session --open" >&2
    return 1
  fi
}

# Read KEY=VALUE from repo .env (comments/blank lines ignored). Does not export secrets
# unless the caller assigns the returned value.
dotenv_get() {
  local file="$1"
  local key="$2"
  local line
  [[ -f "$file" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" ]] || continue
    if [[ "$line" == "$key="* ]]; then
      local value="${line#"$key="}"
      value="${value#\"}"
      value="${value%\"}"
      value="${value#\'}"
      value="${value%\'}"
      printf '%s' "$value"
      return 0
    fi
  done <"$file"
  return 1
}

# Inject env vars into a script before colab exec (used for git clone).
# the target script in a temp file that sets os.environ before the real script runs.
colab_exec_script() {
  local cli="$1"
  local session="$2"
  local timeout="$3"
  local script="$4"
  shift 4

  local tmp
  tmp=$(mktemp "${TMPDIR:-/tmp}/vr-colab-exec.XXXXXX.py")
  {
    echo "import os"
    local pair name value
    for pair in "$@"; do
      name="${pair%%=*}"
      value="${pair#*=}"
      [[ -n "$name" && -n "$value" ]] || continue
      python3 - "$name" "$value" <<'PY'
import json, sys
name, value = sys.argv[1], sys.argv[2]
print(f"os.environ[{json.dumps(name)}] = {json.dumps(value)}")
PY
    done
    cat "$script"
  } >"$tmp"
  "$cli" exec -s "$session" -f "$tmp" --timeout "$timeout"
  local status=$?
  rm -f "$tmp"
  return "$status"
}

colab_ensure_remote_dir() {
  local cli="$1"
  local session="$2"
  local remote_dir="$3"
  printf 'import os\nos.makedirs(%s, exist_ok=True)\n' \
    "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" _ "$remote_dir")" \
    | "$cli" exec -s "$session" --timeout 30 >/dev/null
}

colab_upload_file() {
  local cli="$1"
  local session="$2"
  local local_path="$3"
  local remote_path="$4"
  local remote_dir

  if "$cli" upload -s "$session" "$local_path" "$remote_path"; then
    return 0
  fi

  remote_dir="${remote_path%/*}"
  if [[ "$remote_dir" == "$remote_path" || "$remote_dir" == "/content" ]]; then
    echo "[setup] Upload failed for $remote_path (no retry path)."
    return 1
  fi

  echo "[setup] Upload failed — creating remote dir $remote_dir and retrying..."
  colab_ensure_remote_dir "$cli" "$session" "$remote_dir"
  "$cli" upload -s "$session" "$local_path" "$remote_path"
}

# Build KEY=VALUE pairs from a dotenv file for colab_exec_script.
colab_env_args_from_file() {
  local file="$1"
  shift
  local key value
  COLAB_ENV_ARGS=()
  [[ -f "$file" ]] || return 0
  for key in "$@"; do
    value="$(dotenv_get "$file" "$key" || true)"
    if [[ -n "$value" ]]; then
      COLAB_ENV_ARGS+=("$key=$value")
    fi
  done
}

# Write a local file to a path on the CLI VM (reliable vs colab upload into subfolders).
colab_write_remote_file() {
  local cli="$1"
  local session="$2"
  local remote_path="$3"
  local local_path="$4"
  local timeout="${5:-60}"
  local tmp

  if [[ ! -f "$local_path" ]]; then
    echo "Local file not found: $local_path" >&2
    return 1
  fi

  tmp=$(mktemp "${TMPDIR:-/tmp}/vr-colab-write.XXXXXX.py")
  python3 - "$local_path" "$remote_path" <<'PY' >"$tmp"
import json, pathlib, sys
local, remote = sys.argv[1], sys.argv[2]
content = pathlib.Path(local).read_text(encoding="utf-8")
script = f"""import pathlib
path = pathlib.Path({json.dumps(remote)})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text({json.dumps(content)}, encoding="utf-8")
print(f"Wrote {{path}} ({{len({json.dumps(content)})}} bytes)")
"""
print(script)
PY
  "$cli" exec -s "$session" -f "$tmp" --timeout "$timeout"
  local status=$?
  rm -f "$tmp"
  return "$status"
}
