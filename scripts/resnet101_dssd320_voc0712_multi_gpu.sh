# dataset path
export VOC_ROOT="/data/pzq/voc/VOCdevkit"

# specify gpus
export NGPUS=2
export CUDA_VISIBLE_DEVICES=0,1

# train
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/resnet101_dssd320_voc0712.yaml


# evaluate
# python -m torch.distributed.launch --nproc_per_node=$NGPUS test.py --config-file configs/resnet101_dssd320_voc0712.yaml

