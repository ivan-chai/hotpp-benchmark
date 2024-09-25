for temperature in 10 2 1 0.5 0.1; do
    for method in detection_hybrid; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method}_sample name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.next_item_loss.temperature="${temperature}"
    done
    for method in next_item rmtpp; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.prediction=sample +module.loss.temperature="${temperature}"
    done
    for method in nhp ode attnhp; do
        python3 -m hotpp.evaluate --config-dir configs --config-name ${method} name=${method}_t_${temperature} model_path=checkpoints/${method}.ckpt +module.loss.prediction=sample-labels +module.loss.temperature="${temperature}"
    done
done
