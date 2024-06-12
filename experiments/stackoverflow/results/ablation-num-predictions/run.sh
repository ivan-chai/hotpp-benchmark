for maxpreds in 1 2 4 5 8; do
    python3 -m esp_horizon.evaluate --config-dir configs --config-name next_item ++max_predictions=$maxpreds ++name=next_item_pred_$maxpreds ++model_path=checkpoints/next_item.ckpt '~metric.otd_steps'
    python3 -m esp_horizon.evaluate --config-dir configs --config-name rmtpp ++max_predictions=$maxpreds ++name=rmtpp_pred_$maxpreds ++model_path=checkpoints/rmtpp.ckpt '~metric.otd_steps'
    python3 -m esp_horizon.evaluate --config-dir configs --config-name nhp ++max_predictions=$maxpreds ++name=nhp_pred_$maxpreds ++model_path=checkpoints/nhp.ckpt '~metric.otd_steps'
done
for maxpreds in 12 16 24 32; do
    python3 -m esp_horizon.evaluate --config-dir configs --config-name next_item ++max_predictions=$maxpreds ++name=next_item_pred_$maxpreds ++model_path=checkpoints/next_item.ckpt
    python3 -m esp_horizon.evaluate --config-dir configs --config-name rmtpp ++max_predictions=$maxpreds ++name=rmtpp_pred_$maxpreds ++model_path=checkpoints/rmtpp.ckpt
    python3 -m esp_horizon.evaluate --config-dir configs --config-name nhp ++max_predictions=$maxpreds ++name=nhp_pred_$maxpreds ++model_path=checkpoints/nhp.ckpt
done