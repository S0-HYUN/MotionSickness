# device=0
# ### soso
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --lr=0.001 --test_subj $var --standard=acc --model=soso --criterion=quad --batch_size=16
#     python main_DA_MS.py --device=0 --wd=0.001 --lr=0.001 --test_subj $var --mode=test --standard=acc --model=soso --criterion=quad --batch_size=16
# done

### DeepConvNet
# for var in $(seq 1 8);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=DeepConvNet --criterion=MSE
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=DeepConvNet --criterion=MSE
# done

# ### EEGNet
# for var in $(seq 1 8);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --criterion=MSE
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --mode=test --criterion=MSE
# done

###
for var in $(seq 4 7);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=ShallowConvNet --criterion=MSE
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=ShallowConvNet --criterion=MSE
done




###
# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=1 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=1 --mode=test --standard=loss --class_num=3 --wd=0.001

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=2 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=2 --mode=test --standard=loss --class_num=3 --wd=0.001

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=3 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=3 --mode=test --standard=loss --class_num=3 --wd=0.001

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=4 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=4 --mode=test --standard=loss --class_num=3 --wd=0.001

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=5 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=5 --mode=test --standard=loss --class_num=3 --wd=0.001

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=6 --standard=loss --class_num=3 --wd=0.001
# python main_DA_MS.py --device=0 --test_subj=6 --mode=test --standard=loss --class_num=3 --wd=0.001



# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=1 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=1 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=2 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=2 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=3 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=3 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=4 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=4 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=5 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=5 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 

# python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --test_subj=6 --DA=true --da_epoch=30 --da_lr=1e-6 
# python main_DA_MS.py --device=0 --test_subj=6 --mode=test --DA=true --da_epoch=30 --da_lr=1e-6 



############

### soso
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=acc --model=soso
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=acc --model=soso --mode=test
# done
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=f1 --model=soso
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=f1 --model=soso --mode=test
# done
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=loss --model=soso
#     python main_DA_MS.py --device=0 -sch=step --step_size=10 --gamma=0.5 --lr=0.01 --wd=0.005 --test_subj $var --standard=loss --model=soso --mode=test
# done