for maxpreds in 1 2 4 5 8; do
    for method in next_item next_item_transformer rmtpp nhp ode detection detection_hybrid diffusion_gru; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} ++max_predictions=$maxpreds ++name=${method}_pred_$maxpreds ++model_path=checkpoints/${method}.ckpt '~metric.otd_steps'
    done
done
for maxpreds in 12 16 24 32; do
    for method in next_item next_item_transformer rmtpp nhp ode detection detection_hybrid diffusion_gru; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} ++max_predictions=$maxpreds ++name=${method}_pred_$maxpreds ++model_path=checkpoints/${method}.ckpt
    done
done
