import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = '/home/fengyouliang/datasets/WHD/kaggle_csv/train_0618.csv'
df = pd.read_csv(csv_path)

# get a df with just image and source columns
# such that dropping duplicates will only keep unique image_ids
image_source = df[['image_id', 'source']].drop_duplicates()

# get lists for image_ids and sources
image_ids = image_source['image_id'].to_numpy()
sources = image_source['source'].to_numpy()

ret = train_test_split(image_ids, sources, test_size=0.2, stratify=sources, random_state=None)
X_train, X_test, y_train, y_test = ret

print(f'# train images: {len(X_train)}')
print(f'# val images: {len(X_test)}')

train_df = df[df['image_id'].isin(X_train)]
val_df = df[df['image_id'].isin(X_test)]

fig = plt.figure(figsize=(10, 15))
counts = train_df['source'].value_counts()
ax1 = fig.add_subplot(2, 1, 1)
a = ax1.bar(counts.index, counts)
counts = val_df['source'].value_counts()
ax2 = fig.add_subplot(2, 1, 2)
a = ax2.bar(counts.index, counts)
plt.show()
