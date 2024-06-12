for method in next_k rmtpp_k; do
    for k in 1 2 3 4; do
        python3 -m hotpp.train --config-dir configs --config-name $method ++max_predictions=$k ++name=${method}_$k '~metric.otd_steps'
    done

    for k in 5 8 12 16; do
        python3 -m hotpp.train --config-dir configs --config-name $method ++max_predictions=$k ++name=${method}_$k
    done
done
