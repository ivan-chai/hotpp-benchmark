for maxpreds in 1 2 4; do
    python3 -m hotpp.evaluate --config-dir configs --config-name next_item ++max_predictions=$maxpreds ++name=next_item_pred_$maxpreds ++model_path=checkpoints/next_item.ckpt '~metric.otd_steps'
    python3 -m hotpp.evaluate --config-dir configs --config-name rmtpp ++max_predictions=$maxpreds ++name=rmtpp_pred_$maxpreds ++model_path=checkpoints/rmtpp.ckpt '~metric.otd_steps'
done
for maxpreds in 5 8 12 16; do
    python3 -m hotpp.evaluate --config-dir configs --config-name next_item ++max_predictions=$maxpreds ++name=next_item_pred_$maxpreds ++model_path=checkpoints/next_item.ckpt
    python3 -m hotpp.evaluate --config-dir configs --config-name rmtpp ++max_predictions=$maxpreds ++name=rmtpp_pred_$maxpreds ++model_path=checkpoints/rmtpp.ckpt
done
for maxpreds in 1 2 4; do
    python3 -m hotpp.evaluate --config-dir configs --config-name nhp ++max_predictions=$maxpreds ++name=nhp_pred_$maxpreds ++model_path=checkpoints/nhp.ckpt '~metric.otd_steps'
    python3 -m hotpp.evaluate --config-dir configs --config-name ode ++max_predictions=$maxpreds ++name=ode_pred_$maxpreds ++model_path=checkpoints/ode.ckpt '~metric.otd_steps'
done
for maxpreds in 5 8 12 16; do
    python3 -m hotpp.evaluate --config-dir configs --config-name nhp ++max_predictions=$maxpreds ++name=nhp_pred_$maxpreds ++model_path=checkpoints/nhp.ckpt
    python3 -m hotpp.evaluate --config-dir configs --config-name ode ++max_predictions=$maxpreds ++name=ode_pred_$maxpreds ++model_path=checkpoints/ode.ckpt
done
