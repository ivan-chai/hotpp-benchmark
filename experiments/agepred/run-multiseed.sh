for method in next_item next_k rmtpp rmtpp_k hypro_rmtpp; do
    python3 -m esp_horizon.train_multiseed --config-dir configs --config-name ${method}
done
