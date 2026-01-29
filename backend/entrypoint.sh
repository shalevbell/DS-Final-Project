#!/bin/bash
# Do not use set -e: we must always reach exec "$@" so the app starts even if download fails

# If DRIVE_DATASET_FOLDER_ID is set and /data/dataset is empty (or REFRESH_DATASET=1), download from Google Drive
if [ -n "${DRIVE_DATASET_FOLDER_ID}" ]; then
  if [ ! -d /data/dataset ] || [ -z "$(ls -A /data/dataset 2>/dev/null)" ] || [ "${REFRESH_DATASET}" = "1" ]; then
    echo "[entrypoint] Downloading dataset from Google Drive into /data/dataset (folder ID: ${DRIVE_DATASET_FOLDER_ID})..."
    mkdir -p /data/dataset
    if gdown --folder "https://drive.google.com/drive/folders/${DRIVE_DATASET_FOLDER_ID}" -O /data/dataset --remaining-ok 2>/dev/null; then
      # gdown creates a subdir inside /data/dataset (e.g. Savee-Classifier); move its contents up to /data/dataset
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
        echo "[entrypoint] Dataset ready in /data/dataset ($(ls -A /data/dataset 2>/dev/null | wc -l) items)"
      else
        echo "[entrypoint] WARNING: No subdirectory found under /data/dataset"
      fi
    else
      echo "[entrypoint] WARNING: gdown failed. Starting app anyway. Ensure Drive folder is 'Anyone with the link can view' if you need the dataset."
    fi
  fi
fi

exec "$@"
