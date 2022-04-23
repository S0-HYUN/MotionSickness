# # device=2

### soso
for var in $(seq 13 18);
do 
    python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=acc --model=soso
    python main_DA_MS.py --device=2 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=acc --model=soso
done
for var in $(seq 13 18);
do 
    python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=f1 --model=soso
    python main_DA_MS.py --device=2 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=f1 --model=soso
done
for var in $(seq 13 18);
do 
    python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.05 --lr=0.001 --test_subj $var --standard=loss --model=soso
    python main_DA_MS.py --device=2 --wd=0.05 --lr=0.001 --test_subj $var --mode=test --standard=loss --model=soso
done


# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=DeepConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=DeepConvNet
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=DeepConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=DeepConvNet
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=DeepConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=DeepConvNet
# done

# ###
# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=acc --model=EEGNet --mode=test
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=f1 --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=f1 --model=EEGNet --mode=test
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=loss --model=EEGNet
#     python main_DA_MS.py --device=0 -sch=step --step_size=5 --gamma=0.8 --lr=0.01 --wd=0.01 --optimizer=AdamW --test_subj $var --standard=loss --model=EEGNet --mode=test
# done

###
# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=ShallowConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=ShallowConvNet
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=ShallowConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=ShallowConvNet
# done

# for var in $(seq 13 18);
# do 
#     python main_DA_MS.py --device=2 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=ShallowConvNet
#     python main_DA_MS.py --device=2 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=ShallowConvNet
# done