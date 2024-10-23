for layer in $(seq 0 26); do
    python3 -m hotpp.eval_downstream --config-dir configs --config-name downstream +model_config=llm ++num_workers=12 +model_overrides='["+embeddings_cache=/home/ivan/Works/transllm/2024-10-23-embeddings.cache", "+embeddings_layer='${layer}'", "name=llm-layer-'${layer}'"]'
done
