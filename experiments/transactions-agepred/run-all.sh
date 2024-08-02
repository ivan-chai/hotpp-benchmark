for method in most_popular most_popular_otd recent_history recent_history_otd next_item next_k rmtpp rmtpp_k nhp ode; do
    python3 -m hotpp.train_multiseed --config-dir configs --config-name ${method}
done
for method in next_item next_k rmtpp rmtpp_k nhp ode; do
    python3 -m hotpp.eval_downstream --config-dir configs --config-name downstream +model_config=${method} ++num_workers=8
done
