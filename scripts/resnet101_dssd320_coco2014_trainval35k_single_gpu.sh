# dataset path
export VOC_ROOT="/data/pzq/coco"


# train
python train.py --config-file ./configs/resnet101_dssd320_coco2014_trainval35k.yaml


# evaluate
# python test.py --config-file configs/resnet101_dssd320_coco2014_trainval35k.yaml


# inference
# python demo.py --config-file configs/resnet101_dssd320_coco2014_trainval35k.yaml --images_dir demo --ckpt [ckpt_path]
