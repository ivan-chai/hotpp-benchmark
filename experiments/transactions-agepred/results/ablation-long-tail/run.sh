for method in next_item rmtpp next_k; do
    for top in 4 8 16 32 64 128 203; do
        python3 -m hotpp.evaluate --config-dir configs --config-name $method ++model_path=checkpoints/${method}.ckpt ++name=${method}_top_${top} +metric.top_classes="$top" +data_module.top_classes="$top"
    done
done
