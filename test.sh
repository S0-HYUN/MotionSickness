for var in $(seq 13 15);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj $var --standard=f1 --class_num=3 --wd=0.001 --epoch=2
    python main_DA_MS.py --device=0 --test_subj $var --mode=test --standard=f1 --class_num=3 --wd=0.001
done
