#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: ./run.sh [path_to_mri_image]"
  exit 1
fi

IMAGE_PATH=$1


if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi


source venv/bin/activate


echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt


if [ ! -f "models/pretrained_model.pt" ]; then
  echo "Downloading pretrained model..."
  mkdir -p models
  wget -O models/pretrained_model.pt "YOUR_MODEL_DOWNLOAD_URL"
fi


echo "Running inference on $IMAGE_PATH..."
python src/inference.py --image_path "$IMAGE_PATH"


deactivate
echo "Done!"
