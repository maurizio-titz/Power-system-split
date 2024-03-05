import sys
sys.path.append("./")
from utils.config import path_to_indicator_vectors
from kmeans_clust import load_indicator_vectors, create_component_indicator_vectors
from tqdm import tqdm

co2l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
n_nodes = 400
type = "rocof"

for co2 in tqdm(co2l):
    create_component_indicator_vectors(400, co2, type, path_to_indicator_vectors, mask=None)