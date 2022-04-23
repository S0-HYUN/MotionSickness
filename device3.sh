# # device=1
### soso
for var in $(seq 19 23);
do 
    python main_DA_MS.py --device=3 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --lr=0.001 --test_subj $var --standard=acc --model=soso --criterion=quad --batch_size=16
    python main_DA_MS.py --device=3 --wd=0.001 --lr=0.001 --test_subj $var --mode=test --standard=acc --model=soso --criterion=quad --batch_size=16
done

# ### DeepConvNet
# for var in $(seq 19 23);
# do 
#     python main_DA_MS.py --device=3 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=DeepConvNet --criterion=MSE
#     python main_DA_MS.py --device=3 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=DeepConvNet --criterion=MSE
# done

# ### EEGNet
# for var in $(seq 19 23);
# do 
#     python main_DA_MS.py --device=3 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --criterion=MSE
#     python main_DA_MS.py --device=3 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --mode=test --criterion=MSE
# done


# ### ShallowConvNet
# for var in $(seq 19 23);
# do 
#     python main_DA_MS.py --device=3 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=ShallowConvNet --criterion=MSE
#     python main_DA_MS.py --device=3 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=ShallowConvNet --criterion=MSE
# done