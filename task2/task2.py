import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt


def text_preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

directory = "./lad_academy/task2/sampled_texts"
data = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
            data.append({'texts':text_preprocess(text)})
            
data = pd.DataFrame(data)


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors = 'pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

data['embeddings'] = data['texts'].apply(lambda x: get_embedding(x).numpy())

X = np.vstack(data['embeddings'].values)

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.xticks(k_values)
plt.grid()
plt.show()

silhouette_scores = []

for k in k_values[1:]:
    kmeans = KMeans(n_clusters=k, random_state=10)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X ,cluster_labels)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(k_values[1:], silhouette_scores, marker='o')
plt.xlabel("k")
plt.ylabel('shihouette score')
plt.xticks(k_values[1:])
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=2, random_state=10) 
kmeans.fit(X)

labels = kmeans.labels_

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='plasma', marker='o')
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

new_directory = './lad_academy/test_texts'
new_data = []
for filename in os.listdir(new_directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(new_directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
            new_data.append({'texts':text_preprocess(text)})

new_data = pd.DataFrame(new_data)

new_data['embeddings'] = new_data['texts'].apply(lambda x: get_embedding(x).numpy())
X_ = np.vstack(new_data['embeddings'].values)
new_label = kmeans.predict(X_)
X_reduced_ = pca.transform(X_)

plt.figure(figsize=(10, 6))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', marker='o')

plt.scatter(X_reduced_[:, 0], X_reduced_[:, 1], c='red', marker='X', s=100, label='Новое значение')

print(f'Новые данные: {X_}, Лейбл: {new_label[0]}')
plt.title('Результаты кластеризации KMeans с новым значением')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.colorbar(label='Cluster Label')
plt.show()
