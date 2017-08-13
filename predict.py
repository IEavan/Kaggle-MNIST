import pandas
import train
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

# Constants
BATCH_SIZE = train.BATCH_SIZE

# Load data
test_data = pandas.read_csv("test.csv")
model = train.CNN_Model()
model.load_state_dict(torch.load("model_params/model"))

# Run model on all examples in test_data
all_predictions = []
for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch = test_data[i:i+BATCH_SIZE].values
    batch_predictions = train.predict(batch, model)
    all_predictions.append(batch_predictions)

# Concatenate predictions and transform into single data frame
predictions = np.concatenate(all_predictions, axis=0).reshape(-1,1)
image_id = np.array([i + 1 for i in range(len(test_data))]).reshape(-1,1)
result = pandas.DataFrame(np.concatenate([image_id, predictions], axis=1),
        columns=["ImageId", "Label"])

# Write data frame to file
result.to_csv(path_or_buf="predictions.csv", index=False)
