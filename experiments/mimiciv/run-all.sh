for method in recent_history recent_history_otd most_popular next_item next_k rmtpp rmtpp_k nhp; do
    python3 -m hotpp.train --config-dir configs --config-name ${method}
done
