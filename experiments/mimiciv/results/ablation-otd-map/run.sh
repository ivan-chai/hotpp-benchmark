for method in next_item rmtpp next_k rmtpp_k nhp ode; do
    for horizon in 1 2 4 8 16 22 28; do
        otd_c=$(echo "scale=2; ${horizon} / 2" | bc -l)
        python3 -m hotpp.evaluate --config-dir configs --config-name $method ++model_path=checkpoints/${method}.ckpt ++name=${method}_h_${horizon} ++metric.map_thresholds="[$horizon]" ++metric.otd_insert_cost="$otd_c" ++metric.otd_delete_cost="$otd_c"
    done
done
