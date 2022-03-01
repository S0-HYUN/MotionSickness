import mat73

def main():
    mat_file = mat73.loadmat('/opt/workspace/MS_DATA_public/VRSA-FR/Physiological signal/EEG/11/subj_5.mat')
    final_array = mat_file['final_array']
    print(final_array.shape)


if __name__ == "__main__" :
    main()