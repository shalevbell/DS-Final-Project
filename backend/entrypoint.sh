#!/bin/bash
# Do not use set -e: we must always reach exec "$@" so the app starts even if download fails

# Flatten dataset: if /data/dataset contains a single subdir (e.g. Savee-Classifier), move its contents up
flatten_dataset() {
  SOURCE_DIR=""
  if [ -d /data/dataset/Savee-Classifier ]; then
    SOURCE_DIR=/data/dataset/Savee-Classifier
  elif [ -d /data/dataset/Emotions ]; then
    SOURCE_DIR=/data/dataset/Emotions
  else
    for d in /data/dataset/*/; do
      if [ -d "$d" ]; then
        SOURCE_DIR="$d"
        break
      fi
    done
  fi
  if [ -n "$SOURCE_DIR" ]; then
    echo "[entrypoint] Moving contents to /data/dataset ..."
    cp -a "${SOURCE_DIR%/}/." /data/dataset/ && rm -rf "${SOURCE_DIR}"
  fi
  echo "[entrypoint] Dataset ready in /data/dataset ($(ls -A /data/dataset 2>/dev/null | wc -l) items)"
}

need_download() {
  [ ! -d /data/dataset ] || [ -z "$(ls -A /data/dataset 2>/dev/null)" ] || [ "${REFRESH_DATASET}" = "1" ]
}

# Only download dataset if ENABLE_TRAINING=1 (dataset is only needed for training, not inference)
# This saves time and bandwidth when running in inference-only mode
if [ "${ENABLE_TRAINING}" != "1" ]; then
  echo "[entrypoint] ENABLE_TRAINING not set to '1', skipping dataset download (inference-only mode)"
  echo "[entrypoint] To enable training: set ENABLE_TRAINING=1 in docker-compose.yml"
# Prefer ZIP download: single file = full content (folder download often gets only first page = 2 subfolders)
elif [ -n "${DRIVE_DATASET_ZIP_ID}" ] && need_download; then
  echo "[entrypoint] Downloading dataset ZIP from Google Drive (file ID: ${DRIVE_DATASET_ZIP_ID})..."
  mkdir -p /data/dataset
  if [ "${REFRESH_DATASET}" = "1" ] && [ -n "$(ls -A /data/dataset 2>/dev/null)" ]; then
    echo "[entrypoint] REFRESH_DATASET=1: clearing old dataset..."
    rm -rf /data/dataset/* /data/dataset/.[!.]* 2>/dev/null || true
  fi
  if gdown "https://drive.google.com/uc?id=${DRIVE_DATASET_ZIP_ID}" -O /data/dataset/dataset.zip --remaining-ok 2>/dev/null; then
    echo "[entrypoint] Extracting ZIP..."
    if unzip -o -q /data/dataset/dataset.zip -d /data/dataset 2>/dev/null; then
      rm -f /data/dataset/dataset.zip
      flatten_dataset
    else
      echo "[entrypoint] WARNING: unzip failed. Leaving dataset.zip in /data/dataset."
    fi
  else
    echo "[entrypoint] WARNING: gdown failed for ZIP. Set file to 'Anyone with the link can view'."
  fi
# Fallback: folder download (may get only partial content – first page of Drive API ≈ 2 subfolders)
elif [ -n "${DRIVE_DATASET_FOLDER_ID}" ] && need_download; then
  echo "[entrypoint] Downloading dataset folder from Google Drive (folder ID: ${DRIVE_DATASET_FOLDER_ID})..."
  echo "[entrypoint] Note: folder download may get only part of the files. For full dataset use DRIVE_DATASET_ZIP_ID with a ZIP file."
  mkdir -p /data/dataset
  if gdown --folder "https://drive.google.com/drive/folders/${DRIVE_DATASET_FOLDER_ID}" -O /data/dataset --remaining-ok 2>/dev/null; then
    flatten_dataset
  else
    echo "[entrypoint] WARNING: gdown failed. Ensure Drive folder is 'Anyone with the link can view'."
  fi
fi

# Generic function to download and extract model from Google Drive
download_model_from_drive() {
  local MODEL_NAME="$1"
  local MODEL_DIR="$2"
  local DRIVE_FILE_ID="$3"
  local CHECK_FILE="$4"     # File to check if model exists (e.g., "vocal_tone_model.pkl")
  local ZIP_NAME="$5"       # Name of the ZIP file (e.g., "vocal_tone_model.zip")
  local SKIP_IF_ENV_SET="$6"  # Optional: skip if this env var is set (e.g., VOCAL_TONE_MODEL_DIR)

  echo "[entrypoint] ${MODEL_NAME}: dir=${MODEL_DIR}, DRIVE_ID=${DRIVE_FILE_ID:-<not set>}, has_model=$([ -f "${MODEL_DIR}/${CHECK_FILE}" ] && echo yes || echo no)"

  # Skip if environment override is set
  if [ -n "${SKIP_IF_ENV_SET}" ]; then
    echo "[entrypoint] ${MODEL_NAME}: Using custom model directory (${SKIP_IF_ENV_SET} is set), skipping download."
    return 0
  fi

  # Skip if DRIVE_FILE_ID not provided
  if [ -z "${DRIVE_FILE_ID}" ]; then
    echo "[entrypoint] ${MODEL_NAME}: DRIVE_FILE_ID not set, will use local files if available."
    return 0
  fi

  # Skip if model already exists
  if [ -f "${MODEL_DIR}/${CHECK_FILE}" ]; then
    echo "[entrypoint] ${MODEL_NAME}: Model already present, skipping download."
    return 0
  fi

  # Download and extract model
  echo "[entrypoint] ${MODEL_NAME}: Downloading from Google Drive (file ID: ${DRIVE_FILE_ID})..."
  mkdir -p "${MODEL_DIR}"

  if gdown "https://drive.google.com/uc?id=${DRIVE_FILE_ID}" -O "${MODEL_DIR}/${ZIP_NAME}" --remaining-ok; then
    echo "[entrypoint] ${MODEL_NAME}: Extracting..."
    if unzip -o -q "${MODEL_DIR}/${ZIP_NAME}" -d "${MODEL_DIR}"; then
      rm -f "${MODEL_DIR}/${ZIP_NAME}"
      echo "[entrypoint] ${MODEL_NAME}: Model ready in ${MODEL_DIR}"
    else
      echo "[entrypoint] WARNING: ${MODEL_NAME}: unzip of ${ZIP_NAME} failed."
    fi
  else
    echo "[entrypoint] WARNING: ${MODEL_NAME}: gdown failed. Set file to 'Anyone with the link can view'."
  fi
}

# Download Vocal Tone model
VOCAL_TONE_DIR="${VOCAL_TONE_MODEL_DIR:-/app/vocal_tone_model/models}"
download_model_from_drive \
  "Vocal Tone" \
  "${VOCAL_TONE_DIR}" \
  "${DRIVE_VOCAL_TONE_MODEL_ZIP_ID}" \
  "vocal_tone_model.pkl" \
  "vocal_tone_model.zip" \
  "${VOCAL_TONE_MODEL_DIR}"

# Download Clifton Fusion model
CLIFTON_MODEL_DIR="/app/clifton_model/models"
download_model_from_drive \
  "Clifton Fusion" \
  "${CLIFTON_MODEL_DIR}" \
  "${DRIVE_CLIFTON_MODEL_ZIP_ID}" \
  "clifton_fusion_model.pkl" \
  "clifton_model.zip" \
  ""

exec "$@"
