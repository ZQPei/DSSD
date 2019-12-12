# dataset path
export VOC_ROOT="/data/pzq/voc/VOCdevkit"


# train
python train.py --config-file ./configs/resnet101_dssd320_voc0712.yaml


# evaluate
# python test.py --config-file configs/resnet101_dssd320_voc0712.yaml


# inference
# python demo.py --config-file configs/resnet101_dssd320_voc0712.yaml --images_dir demo --ckpt [ckpt_path]
