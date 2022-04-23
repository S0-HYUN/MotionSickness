# device=0
### soso
for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=acc --model=soso
    python main_DA_MS.py --device=0 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=acc --model=soso
done
for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=f1 --model=soso
    python main_DA_MS.py --device=0 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=f1 --model=soso
done
for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=loss --model=soso
    python main_DA_MS.py --device=0 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=loss --model=soso
done

### DeepConvNet
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=DeepConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=DeepConvNet
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=DeepConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=DeepConvNet
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=DeepConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=DeepConvNet
# done

###
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --mode=test
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=f1 --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=f1 --model=EEGNet --mode=test
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=loss --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=loss --model=EEGNet --mode=test
# done

###
# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=ShallowConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=ShallowConvNet
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=ShallowConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=ShallowConvNet
# done

# for var in $(seq 1 6);
# do 
#     python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=ShallowConvNet
#     python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=ShallowConvNet
# done




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