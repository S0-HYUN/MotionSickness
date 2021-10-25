#!/bin/sh

for var in $(seq 1 9); #(10 23)
do
    echo $var
    python main.py --test_subj $var
done

# python main.py --test_list 10
# python main.py --test_list 11
# python main.py --test_list 12
# python main.py --test_list 13