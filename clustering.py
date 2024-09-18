from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Any


def cluster_texts(tagged_texts: List[Dict[str, Any]], n_clusters: int = 5):
    # Collect all meta tags for each text
    meta_tags = [' '.join(item.get('All Meta Tags', []))
                 for item in tagged_texts]
    # Use TF-IDF vectorizer on meta tags
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(meta_tags)
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    # Assign cluster labels
    for idx, item in enumerate(tagged_texts):
        item['Cluster'] = int(kmeans.labels_[idx])
    return tagged_texts
