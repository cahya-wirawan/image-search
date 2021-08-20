import pickle

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer
import datasets

# Used to create and store the Faiss index.
import faiss
import numpy as np

class WitIndex:
    """
    WitIndex is a class to search the wiki snippets from the given text. It can also return link to the
    wiki page or the image.
    """
    wit_dataset = None

    def __init__(self, wit_index_path: str, model_name: str, wit_dataset_path: str, gpu=True):
        self.index = faiss.read_index(wit_index_path)
        self.model = SentenceTransformer(model_name)
        if WitIndex.wit_dataset is None:
            WitIndex.wit_dataset = pickle.load(open(wit_dataset_path, "rb"))
        if gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def search(self, text, top_k=6):
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        # Retrieve the k nearest neighbours
        distance, index = self.index.search(np.array([embedding]), k=top_k)
        distance, index = distance.flatten().tolist(), index.flatten().tolist()
        index_url = [WitIndex.wit_dataset['desc2image_map'][i] for i in index]
        image_info = [WitIndex.wit_dataset['image_info'][i] for i in index_url]
        return distance, index, image_info
