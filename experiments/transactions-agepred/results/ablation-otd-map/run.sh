for method in next_item next_item_transformer rmtpp next_k rmtpp_k nhp ode detection detection_hybrid diffusion_gru; do
    for horizon in 0.1 0.2 0.5 1 2 3 4 5 6 7; do
        otd_c=$(echo "scale=2; ${horizon} / 2" | bc -l)
        python3 -m hotpp.evaluate --config-dir configs --config-name $method ++model_path=checkpoints/${method}.ckpt ++name=${method}_h_${horizon} ++metric.map_deltas="[$horizon]" ++metric.otd_insert_cost="$otd_c" ++metric.otd_delete_cost="$otd_c"
    done
done
