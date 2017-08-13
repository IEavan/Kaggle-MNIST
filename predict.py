import pandas
import train
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

BATCH_SIZE = train.BATCH_SIZE
data_frame = pandas.read_csv("test.csv")
model = train.CNN_Model()
model.load_state_dict(torch.load("model_params/model"))

all_predictions = []
for i in tqdm(range(0, len(data_frame), BATCH_SIZE)):
    batch = data_frame[i:i+BATCH_SIZE].values
    batch_predictions = train.predict(batch, model)
    all_predictions.append(batch_predictions)
predictions = np.concatenate(all_predictions, axis=0).reshape(-1,1)
image_id = np.array([i + 1 for i in range(len(data_frame))]).reshape(-1,1)
result = pandas.DataFrame(np.concatenate([image_id, predictions], axis=1),
        columns=["ImageId", "Label"])
result.to_csv(path_or_buf="predictions.csv", index=False)
