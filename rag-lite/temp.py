from helpers import load_config

for x in ['inference_model', 'embedding_model', 'reranker_model']:
    print(f"value of {x}: {load_config(x)}")