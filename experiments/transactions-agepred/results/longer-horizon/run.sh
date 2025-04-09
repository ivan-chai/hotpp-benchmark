set -e
for k in 1 2 3 4; do
    max_predictions=$(( 32 * $k ))
    horizon=$(( 7 * $k ))
    otd=$(( 5 * $k ))
    for method in next_item next_item_transformer rmtpp nhp ode; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} metric.otd_steps=$otd metric.horizon=$horizon max_predictions=$max_predictions name=${method}_longer_$k model_path=checkpoints/${method}.ckpt
    done
done
for k in 1 2 3 4; do
    max_predictions=$(( 32 * $k ))
    horizon=$(( 7 * $k ))
    otd=$(( 5 * $k ))
    for method in detection_hybrid diffusion_gru; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} metric.otd_steps=$otd metric.horizon=$horizon max_predictions=$max_predictions name=${method}_longer_$k model_path=checkpoints/${method}.ckpt data_module.batch_size=16 +module.recurrent_steps=$k
    done
done
