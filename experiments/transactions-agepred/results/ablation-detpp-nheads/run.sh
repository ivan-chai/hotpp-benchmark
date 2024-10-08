for nheads in 48 64 96 128; do
    python3 -m hotpp.train --config-dir configs --config-name detection name=detection_k_${nheads} model_path=checkpoints/detection.ckpt detection_k=${nheads} "~logger"
done
