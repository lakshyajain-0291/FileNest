import os
import json
from sklearn.cluster import KMeans
from helpers.helpers import load_texts, form_cluster_embed_map
from Models import nomic_text
import zmq

TEXT_FOLDER = r"ai/files/txt_files"
BOOTSTRAP_CLUSTERS = 3  # Tune as per requirement
OUTPUT_FOLDER = "clustered_output"

texts, fnames, metadatas = load_texts(TEXT_FOLDER)
embeddings = nomic_text.text_embedder(texts)

kmeans = KMeans(n_clusters=BOOTSTRAP_CLUSTERS,
                n_init="auto", random_state=42)

# cluster_id are assigned cluster numbers for each embed.
cluster_ids = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_.tolist()

cluster_embed_map = form_cluster_embed_map(embeddings, cluster_ids)

cluster_data = []  # This will store a list of all cluster_info

# Forming the appropriate JSON
for cluster_id in cluster_embed_map:
    cluster_info = {
        "centroid": centroids[cluster_id],
        "files": []
    }
    for idx, label in enumerate(cluster_ids):
        if label == cluster_id:

            cluster_info["files"].append({
                "filename": fnames[idx],
                "metadata": metadatas[idx],
                "embedding": embeddings[idx].tolist()
            })
    cluster_data.append(cluster_info)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(os.path.join(OUTPUT_FOLDER, "cluster_embeddings.json"), "w") as f:
    json.dump({"Clusters": cluster_data}, f, indent=2)

context = zmq.Context()
socket = context.socket(zmq.PUSH)  # Socket of Response Type
socket.bind("tcp://127.0.0.1:5555")  # Socket will communicate to this addr

print("sending json")
socket.send_json({"Clusters": cluster_data})
print("json sent")
print({"Clusters": cluster_data})
