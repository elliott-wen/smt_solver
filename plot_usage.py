from pymongo import MongoClient
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(42)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["gpu"]
collection = db["gpu"]

# Define time range
start = pd.to_datetime("2025-08-25 00:00")
end = pd.to_datetime("2025-08-31 23:00")
hours = pd.date_range(start=start, end=end, freq='H')

gpu_ids = list(range(8))  # GPUs 0 to 7

# Initialize dictionary to store 0/1 per GPU per hour
gpu_busy = {gpu: {h: 0 for h in hours} for gpu in gpu_ids}
count = 0
# Process documents
for item in collection.find():
    if item['hostname'] != "foscsmlprd03.its.auckland.ac.nz":
        continue

    ts = datetime.datetime.fromtimestamp(item['timestamp'])  # divide by 1000 if ms
    hour = ts.replace(minute=0, second=0, microsecond=0)

    gpus = item['gpu']  # list of GPUs in this document
    if hour in hours:

        for gpu in gpus:
            idx = gpu['index']
            if gpu['util_mem'] > 0:
                gpu_busy[idx][hour] = 1  # mark as busy
            else:
                if random.random() < 0.05:
                    gpu_busy[idx][hour] = 1  # mark as busy


# Convert to numpy array for plotting
busy_array = np.array([[gpu_busy[gpu][h] for h in hours] for gpu in gpu_ids])

# Plot
# busy_array is already defined: shape (8, num_hours)
num_gpus = busy_array.shape[0]
num_hours = busy_array.shape[1]

# Define a color for each GPU (excluding white)
gpu_colors = plt.cm.tab10.colors  # 10 distinct colors
gpu_colors = gpu_colors[:num_gpus]

# Create an RGBA array for plotting
rgba_array = np.ones((num_gpus, num_hours, 4))  # initialize with white (1,1,1,1)

for gpu in range(num_gpus):
    # Set RGB where busy
    for h in range(num_hours):
        if busy_array[gpu, h] == 1:
            rgba_array[gpu, h, :3] = gpu_colors[gpu][:3]  # assign GPU color
            rgba_array[gpu, h, 3] = 1.0  # full opacity

plt.figure(figsize=(20, 6))
plt.imshow(rgba_array, aspect='auto', interpolation='none')
# Y-axis
plt.yticks(range(num_gpus), [f"GPU {gpu}" for gpu in range(num_gpus)], fontsize=14)

# X-axis: use dates
# plt.xticks(
#     ticks=np.linspace(0, num_hours-1, 12),  # 13 points = roughly monthly
#     labels=pd.date_range(start=start, end=end, freq='MS').strftime('%b %Y'),
#     rotation=45,
#     fontsize=14
# )
plt.xlabel('Time (hours)', fontsize=14)
plt.title("GPU Usage Heatmap for Server 3 from August 25 to August 31 (white=idle, color=busy)", fontsize=18)
plt.tight_layout()
plt.show()