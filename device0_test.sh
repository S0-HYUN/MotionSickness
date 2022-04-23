# device=0

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=DeepConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=DeepConvNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=DeepConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=DeepConvNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=DeepConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=DeepConvNet --epoch=2
done

###
for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=EEGNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=EEGNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=EEGNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=EEGNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=EEGNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=EEGNet --epoch=2
done

###
for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=acc --model=ShallowConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=acc --model=ShallowConvNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=f1 --model=ShallowConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=f1 --model=ShallowConvNet --epoch=2
done

for var in $(seq 1 6);
do 
    python main_DA_MS.py --device=0 -sch=cosine --T_max=99 --eta_min=0 --wd=0.001 --test_subj $var --standard=loss --model=ShallowConvNet --epoch=2
    python main_DA_MS.py --device=0 --wd=0.001 --test_subj $var --mode=test --standard=loss --model=ShallowConvNet --epoch=2
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