# for var in $(seq 1 23); #(10 23)


for var in 1 5 9 13 17; # 1 5 9 13 17
do
    echo $((var+1))
    python main_DA_MS.py --device=0 --mode=test --test_subj=$((var)) --DA=true --da_epoch=30 --da_lr=1e-6 &
    python main_DA_MS.py --device=1 --mode=test --test_subj=$((var+1)) --DA=true --da_epoch=30 --da_lr=1e-6 &
    python main_DA_MS.py --device=2 --mode=test --test_subj=$((var+2)) --DA=true --da_epoch=30 --da_lr=1e-6 &
    python main_DA_MS.py --device=3 --mode=test --test_subj=$((var+3)) --DA=true --da_epoch=30 --da_lr=1e-6 &
    # python main_eegnet.py --test_subj $var -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=1e-4 --epoch=3000 --wd=1e-3
    # python main_eegnet.py --test_subj $var -sch=cosine --T_max=1000 --eta_min=0.999 --mode=train --lr=1e-2
    # python main_eegnet.py --test_subj $var -sch=exp --gamma=0.995 --test_subj=18 --lr=1e-4 --wd=1e-3 
done

python main_DA_MS.py --device=0 --mode=test --test_subj=21 --DA=true --da_epoch=30 --da_lr=1e-6 &
python main_DA_MS.py --device=1 --mode=test --test_subj=22 --DA=true --da_epoch=30 --da_lr=1e-6 &
python main_DA_MS.py --device=2 --mode=test --test_subj=23 --DA=true --da_epoch=30 --da_lr=1e-6


# do
#     echo $var
#     python main_DA_MS.py --device=3 -sch=cosine --T_max=99 --eta_min=0 --DA=true --da_epoch=30 --da_lr=1e-6 --test_subj $var --mode=test &
# done




# python main_DA_MS.py --device=0 --test_subj=1 --DA=true --da_epoch=30 --da_lr=1e-6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=0 --test_subj=2 --DA=true --da_epoch=30 --da_lr=1e-6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=0 --test_subj=3 --DA=true --da_epoch=30 --da_lr=1e-6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=0 --test_subj=4 --DA=true --da_epoch=30 --da_lr=1e-6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=0 --test_subj=5 --DA=true --da_epoch=30 --da_lr=1e-6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=0 --test_subj=6 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=7 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=8 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=9 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=10 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=11 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=1 --test_subj=12 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=13 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=14 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=15 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=16 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=17 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=2 --test_subj=18 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=3 --test_subj=19 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=3 --test_subj=20 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=3 --test_subj=21 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=3 --test_subj=22 -sch=cosine --T_max=99 --eta_min=0 &
# python main_DA_MS.py --device=3 --test_subj=23 -sch=cosine --T_max=99 --eta_min=0 