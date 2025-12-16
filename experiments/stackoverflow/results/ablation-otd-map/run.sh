BASE_HORIZON=$(cat configs/default.yaml | grep ' horizon:' | sed 's/#.*//;s/.*://' | tr -d ' ')
BASE_DELTA=$(cat configs/default.yaml | grep ' map_deltas:' | sed 's/#.*//;s/.*://' | tr -d ' []')
BASE_COST=$(cat configs/default.yaml | grep ' otd_insert_cost:' | sed 's/#.*//;s/.*://' | tr -d ' ')
for method in next_item diffusion_gru next_item_transformer rmtpp nhp hypro_rmtpp; do
    if [[ $method == "hypro_rmtpp" ]]; then
        n_seeds=1
    else
        n_seeds=3
    fi
    for scale in 0.1 0.2 0.5 1 2; do
        otd_c=$(echo "${BASE_COST} * ${scale}" | bc -l)
        horizon=$(echo "${BASE_HORIZON} * ${scale}" | bc -l)
        delta="${BASE_DELTA}"
        name=${method}_h_${horizon}_c_${otd_c}
        python3 -m hotpp.evaluate_multiseed --config-dir configs --config-name $method ++model_path=checkpoints/${method}.ckpt name=$name metric.map_deltas="[$delta]" metric.horizon="$horizon" metric.otd_insert_cost="$otd_c" metric.otd_delete_cost="$otd_c" num_evaluation_seeds=$n_seeds
    done
done
