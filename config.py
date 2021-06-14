# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~/hdd/Python/210518_SSD_graduate")

CLASSES = ('person', ' ')
# note: if you used our download scripts, this should be right
ROOT = os.path.join(HOME, "data/VOTT/")


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = [104, 117, 123]

SIZE = 300

image_maisu = 684
#image_maisu = 810
#image_maisu = 1320
batch = 8
epoch = 2000
iterations = int((image_maisu / batch) * epoch)
print('iterations is ', iterations)

data_cfg = {
    'num_classes': 2,
    'lr_steps': ((iterations * 6) / 8, (iterations * 7) / 8, (iterations * 8) / 8),
    'max_iter': iterations,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOTT',
}



mb1_cfg = {
    'num_classes': 2,
    'lr_steps': ((iterations * 6) / 8, (iterations * 7) / 8, (iterations * 8) / 8),
    'max_iter': iterations,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 150, 300],
    'min_sizes': [60, 105, 150, 195, 240, 285],
    'max_sizes': [205, 150, 195, 240, 285, 330],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOTT',
}
