#!/bin/bash

./train_gtex_multi_no_bs_random.sh 5506 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5507 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5508 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5509 &
sleep 20
./train_gtex_multi_no_bs_random.sh 5510 &

wait