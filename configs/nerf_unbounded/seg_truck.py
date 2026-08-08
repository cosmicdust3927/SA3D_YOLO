_base_ = './seg_nerf_unbounded_default.py'

expname = 'truck'

data = dict(
    datadir='./data/360_v2/truck',
    factor=2,
)

yolo = dict(
    target_class_id=7,
)
