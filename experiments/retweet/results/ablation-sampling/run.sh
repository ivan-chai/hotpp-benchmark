for temperature in 10 2 1 0.5 0.1; do
    for method in next_item_transformer; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}-seed-0.ckpt +module.loss.prediction=sample +module.loss.temperature="${temperature}"
    done
    for method in diffusion_gru; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}-seed-0.ckpt +module.loss.next_item_loss.prediction=sample +module.loss.next_item_loss.temperature="${temperature}"
    done
done
# Mean prediction.
temperature=0
for method in next_item_transformer diffusion_gru; do
    python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}-seed-0.ckpt
done
