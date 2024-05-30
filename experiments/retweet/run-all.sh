for method in next_item next_k rmtpp rmtpp_k nhp; do
    python3 -m esp_horizon.train --config-dir configs --config-name ${method}
done
