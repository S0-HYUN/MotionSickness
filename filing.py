def calc_test_file(test_subj, expt):
        if isinstance(test_subj, list) : #type(test_subj) == 'list' :
                test_s = test_subj[0]
        else :
                test_s = test_subj
        test_file = 0
        if expt == 1 :
                test_file = 4 * test_s - 2
        else :
                test_file = 4 * test_s - 1
        return test_file

def naming(df):
        # print("naming에 들어옵니다")
        df.columns = ['Time', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
            'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
            'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
            'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)', 'ECG.(uV)',
            'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)', 'TRIGGER(DIGITAL)']

def use_naming(df):
    # print("use_naming에 들어옵니다.")
    df.columns = ['Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
            'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
            'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
            'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)', 'ECG.(uV)', 'TRIGGER(DIGITAL)']
    
    # df.columns = ['Time', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
    #         'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
    #         'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
    #         'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)', 'ECG.(uV)', 'TRIGGER(DIGITAL)']

if __name__ ==  '__main__' :
        calc_test_file(13, 2)