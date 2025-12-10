import pickle
import numpy as np
import nltk
from nltk.corpus import twitter_samples
from utils import process_tweet, cosine_similarity


print("Loading embeddings...")
en_embeddings_subset = pickle.load(open("en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("fr_embeddings.p", "rb"))

print("Loading tweets...")
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

print(f"Total tweets: {len(all_tweets)}")

def get_document_embedding(tweet, en_embeddings):

    doc_embedding = np.zeros(300)
    processed_tokens = process_tweet(tweet)

    for word in processed_tokens:
        if word in en_embeddings:
            doc_embedding += en_embeddings[word]

    return doc_embedding


def get_document_vecs(all_docs, en_embeddings):
    ind2Doc_dict = {}
    document_vec_l = []

    for i, doc in enumerate(all_docs):
        doc_vec = get_document_embedding(doc, en_embeddings)
        ind2Doc_dict[i] = doc_vec
        document_vec_l.append(doc_vec)

    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict

print("\n--- Testing Part 1: Document Embeddings ---")
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
print(f"Custom tweet embedding (last 5): {tweet_embedding[-5:]}")
# Expected Output: [-0.00268555 -0.15378189 -0.55761719 -0.07216644 -0.32263184]

document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)
print(f"length of dictionary: {len(ind2Tweet)}")
print(f"shape of document_vecs: {document_vecs.shape}")
# Expected Output: length 10000, shape (10000, 300)

N_VECS = len(all_tweets)
N_DIMS = len(ind2Tweet[1])
N_PLANES = 10
N_UNIVERSES = 25

np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)]


def hash_value_of_vector(v, planes):
    dot_product = np.dot(v, planes)

    sign_of_dot_product = np.sign(dot_product)
    h = (sign_of_dot_product >= 0).astype(int)

    h = np.squeeze(h)

    hash_value = 0
    n_planes = planes.shape[1]

    for i in range(n_planes):
        hash_value += (2 ** i) * h[i]

    return int(hash_value)


def make_hash_table(vecs, planes):

    num_of_planes = planes.shape[1]
    num_buckets = 2 ** num_of_planes

    hash_table = {i: [] for i in range(num_buckets)}
    id_table = {i: [] for i in range(num_buckets)}

    for i, v in enumerate(vecs):
        h = hash_value_of_vector(v, planes)

        hash_table[h].append(v)
        id_table[h].append(i)

    return hash_table, id_table


print("\n--- Testing Part 2: Hashing ---")
np.random.seed(0)
idx = 0
planes = planes_l[idx]
vec = np.random.rand(1, 300)
print(f"The hash value for this vector, and the set of planes at index {idx}, is {hash_value_of_vector(vec, planes)}")
# Expected Output: 768

print("Testing make_hash_table (using document_vecs)...")
tmp_hash_table, tmp_id_table = make_hash_table(document_vecs, planes)
print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
print(f"The id table at key 0 has {len(tmp_id_table[0])}")
print(f"The first 5 document indices stored at key 0 are {tmp_id_table[0][0:5]}")
# Expected Output: 3 vectors, 3 ids, indices [3276, 3281, 3282] (output might vary slightly with nltk version but should be close)

# 创建所有 Universes 的 Hash Tables
print("\nCreating hash tables for all universes...")
hash_tables = []
id_tables = []
for universe_id in range(N_UNIVERSES):
    # print('working on hash universe #:', universe_id)
    planes = planes_l[universe_id]
    hash_table, id_table = make_hash_table(document_vecs, planes)
    hash_tables.append(hash_table)
    id_tables.append(id_table)


def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= N_UNIVERSES

    # 1. 收集候选邻居
    vecs_to_consider_l = list()
    ids_to_consider_l = list()

    # 使用集合来去重 (candidate ids)
    ids_to_consider_set = set()

    for universe_id in range(num_universes_to_use):
        planes = planes_l[universe_id]
        hash_value = hash_value_of_vector(v, planes)

        new_ids_to_consider = id_tables[universe_id][hash_value]

        for i in new_ids_to_consider:
            ids_to_consider_set.add(i)

    if doc_id in ids_to_consider_set:
        ids_to_consider_set.remove(doc_id)

    candidate_ids = list(ids_to_consider_set)

    similarity_list = []
    for neighbor_id in candidate_ids:
        neighbor_vec = document_vecs[neighbor_id]
        sim = cosine_similarity(v, neighbor_vec)

        similarity_list.append((neighbor_id, sim))

    similarity_list.sort(key=lambda x: x[1], reverse=True)

    nearest_neighbor_ids = [x[0] for x in similarity_list[:k]]

    return nearest_neighbor_ids


print("\n--- Testing Part 3: Approximate K-NN ---")
doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]

nearest_neighbor_ids = approximate_knn(
    doc_id,
    vec_to_search,
    planes_l,
    k=3,
    num_universes_to_use=5
)

print(f"Nearest neighbors for document {doc_id}")
print(f"Document contents: {doc_to_search}")
print("")

for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {all_tweets[neighbor_id]}")
