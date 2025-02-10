for temperature in 10 2 1 0.5 0.1; do
    for method in detection detection_hybrid rmtpp; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method}_sample name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.next_item_loss.temperature="${temperature}"
    done
    for method in next_item next_item_transformer; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.prediction=sample +module.loss.temperature="${temperature}"
    done
    for method in diffusion_gru; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.next_item_loss.prediction=sample +module.loss.next_item_loss.temperature="${temperature}"
    done
    for method in nhp ode attnhp; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.prediction=sample-labels +module.loss.temperature="${temperature}"
    done
done
# Mean prediction.
temperature=0
for method in detection detection_hybrid next_item next_item_transformer rmtpp nhp ode attnhp diffusion_gru; do
    python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt
done
