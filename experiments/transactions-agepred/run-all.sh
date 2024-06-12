for method in next_item next_k rmtpp rmtpp_k nhp; do
    python3 -m hotpp.train --config-dir configs --config-name ${method}
    python3 -m hotpp.eval_downstream --config-dir configs --config-name downstream +model_config=${method} ++num_workers=8
done
