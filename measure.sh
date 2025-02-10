#!/bin/bash

# 默认参数
MODEL=DiT-XS/2
VAE=mse
IMAGE_SIZE=256
NUM_STEPS=50
SEED=0
IN_PATH=/home/ubuntu/data/datasets/realsr/lr
OUT_PATH=/home/ubuntu/data/datasets/realsr/res
METRICS_PATH=/home/ubuntu/data/datasets/realsr/hr
CKPT=""  # 默认为空，用户可指定路径
export PYTHONWARNINGS="ignore"
# 使用说明
usage() {
  echo "Usage: $0 [-m MODEL] [-v VAE] [-s IMAGE_SIZE] [-n NUM_STEPS] [-d SEED] [-i IN_PATH] [-o OUT_PATH] [-g METRICS_PATH] [-c CUDA_DEVICE] [-p SAMPLING_SCRIPT] [-k CKPT_PATH]"
  echo "Options:"
  echo "  -m MODEL              Model name (default: DiT-XS/2)"
  echo "  -v VAE                VAE type (ema or mse, default: mse)"
  echo "  -s IMAGE_SIZE         Image size (256, 512, or 1024, default: 256)"
  echo "  -n NUM_STEPS          Number of sampling steps (default: 50)"
  echo "  -d SEED               Random seed (default: 0)"
  echo "  -i IN_PATH            Input path for low-quality images (default: $IN_PATH)"
  echo "  -o OUT_PATH           Output path for generated images (default: $OUT_PATH)"
  echo "  -g METRICS_PATH       Ground truth path for metrics (default: $METRICS_PATH)"
  echo "  -c CUDA_DEVICE        CUDA device ID (e.g., 0,1,2)"
  echo "  -p SAMPLING_SCRIPT    Sampling script (sample_ddim.py or sample.py)"
  echo "  -k CKPT_PATH          Optional path to a DiT checkpoint"
  exit 1
}

# 解析命令行参数
while getopts "m:v:s:n:d:i:o:g:c:p:k:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    v) VAE="$OPTARG" ;;
    s) IMAGE_SIZE="$OPTARG" ;;
    n) NUM_STEPS="$OPTARG" ;;
    d) SEED="$OPTARG" ;;
    i) IN_PATH="$OPTARG" ;;
    o) OUT_PATH="$OPTARG" ;;
    g) METRICS_PATH="$OPTARG" ;;
    c) CUDA_DEVICE="$OPTARG" ;;
    p) SAMPLING_SCRIPT="$OPTARG" ;;
    k) CKPT="$OPTARG" ;;
    *) usage ;;
  esac
done

# 检查必要参数
if [ -z "$CUDA_DEVICE" ] || [ -z "$SAMPLING_SCRIPT" ]; then
  echo "Error: CUDA device ID and sampling script are required."
  usage
fi

# 打印配置
echo "===================================="
echo "Configuration:"
echo "Model:               $MODEL"
echo "VAE:                 $VAE"
echo "Image Size:          $IMAGE_SIZE"
echo "Sampling Steps:      $NUM_STEPS"
echo "Seed:                $SEED"
echo "Input Path:          $IN_PATH"
echo "Output Path:         $OUT_PATH"
echo "Ground Truth Path:   $METRICS_PATH"
echo "CUDA Device:         $CUDA_DEVICE"
echo "Sampling Script:     $SAMPLING_SCRIPT"
echo "DiT Checkpoint Path: $CKPT"
echo "===================================="

# 确保输出目录存在
mkdir -p "$OUT_PATH"

# 设置环境变量
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 执行采样
echo "Starting sampling..."
python "$SAMPLING_SCRIPT" \
  --model "$MODEL" \
  --vae "$VAE" \
  --image-size "$IMAGE_SIZE" \
  --num-sampling-steps "$NUM_STEPS" \
  --seed "$SEED" \
  --in-path "$IN_PATH" \
  --out-path "$OUT_PATH" \
  --ckpt "$CKPT" \


# 执行质量测量
echo "Starting quality metrics..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" pyiqa psnry ssim lpips lpips-vgg musiq clipiqa maniqa niqe -t "$OUT_PATH" -r "$METRICS_PATH"

echo "All done!"