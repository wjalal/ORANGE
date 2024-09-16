#!/bin/bash

./train_gtex_multi_no_bs_random.sh 5501 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5502 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5503 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5504 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5505 &

wait