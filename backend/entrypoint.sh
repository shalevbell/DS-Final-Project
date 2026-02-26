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

# Prefer ZIP download: single file = full content (folder download often gets only first page = 2 subfolders)
if [ -n "${DRIVE_DATASET_ZIP_ID}" ] && need_download; then
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

# Vocal Tone model: download from Google Drive if not present (so we don't need to retrain every time)
VOCAL_TONE_DIR="${VOCAL_TONE_MODEL_DIR:-/app/models/vocal_tone}"
echo "[entrypoint] Vocal Tone: dir=${VOCAL_TONE_DIR}, DRIVE_ID=${DRIVE_VOCAL_TONE_MODEL_ZIP_ID:-<not set>}, has_pkl=$([ -f "${VOCAL_TONE_DIR}/vocal_tone_model.pkl" ] && echo yes || echo no)"
if [ -z "${VOCAL_TONE_MODEL_DIR}" ] && [ -n "${DRIVE_VOCAL_TONE_MODEL_ZIP_ID}" ]; then
  if [ ! -f "${VOCAL_TONE_DIR}/vocal_tone_model.pkl" ]; then
    echo "[entrypoint] Downloading Vocal Tone model from Google Drive (file ID: ${DRIVE_VOCAL_TONE_MODEL_ZIP_ID})..."
    mkdir -p "${VOCAL_TONE_DIR}"
    if gdown "https://drive.google.com/uc?id=${DRIVE_VOCAL_TONE_MODEL_ZIP_ID}" -O "${VOCAL_TONE_DIR}/vocal_tone_model.zip" --remaining-ok; then
      echo "[entrypoint] Extracting Vocal Tone model..."
      if unzip -o -q "${VOCAL_TONE_DIR}/vocal_tone_model.zip" -d "${VOCAL_TONE_DIR}"; then
        rm -f "${VOCAL_TONE_DIR}/vocal_tone_model.zip"
        echo "[entrypoint] Vocal Tone model ready in ${VOCAL_TONE_DIR}"
      else
        echo "[entrypoint] WARNING: unzip of vocal_tone_model.zip failed."
      fi
    else
      echo "[entrypoint] WARNING: gdown failed for Vocal Tone model. Set file to 'Anyone with the link can view'. Run: docker-compose logs backend"
    fi
  else
    echo "[entrypoint] Vocal Tone model already present, skipping download."
  fi
fi

exec "$@"
