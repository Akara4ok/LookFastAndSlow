#anchors
FEATURE_MAP_SHAPES = [19, 10, 5, 3, 2, 1]
ASPECT_RATIOS = [[1.0, 2.0, 0.5]] * 6
VARIANCES = [0.1, 0.1, 0.2, 0.2]
IOU_THRESHOLD = 0.5
CONFIDENCE = 0.5
TOP_K_CLASSES = 10

#dataset
DATASET_PATH = "Data/"
TEST_PERCENT = 0.2

#model
MODEL_PATH = "Model/model.weights.h5"
IMG_SIZE = (300, 300, 3)

#train
LOGS_PATH = "Logs/"
SEED = 42
EPOCHS = 30
BATCH_SIZE = 8

#lr
LR = 0.001