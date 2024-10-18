from bikewaysim.paths import config

def handle_directories(subset,calibration_name):
    # handle the directories for exporting results
    if subset is not None:
        first_part = subset[0]
    else:
        first_part = "all"
    
    # format the filepaths
    result_fp = config['calibration_fp'] / f"results/{first_part},{calibration_name},0.pkl"
    routing_fp = config['calibration_fp'] / f"routing/{first_part},{calibration_name},0.pkl"
    loss_fp = config['calibration_fp'] / f"loss/{first_part},{calibration_name},0.pkl"

    # create directories if they don't exist
    if result_fp.parent.exists() == False:
        result_fp.parent.mkdir()
    if routing_fp.parent.exists() == False:
        routing_fp.parent.mkdir()
    if loss_fp.parent.exists() == False:
        loss_fp.parent.mkdir()

    return result_fp, routing_fp, loss_fp

def get_dirctories():
    result_fps = (config['calibration_fp'] / f"results").glob("*.pkl")
    routing_fps = (config['calibration_fp'] / f"routing").glob("*.pkl")
    loss_fps = (config['calibration_fp'] / f"loss").glob("*.pkl")
    return result_fps, routing_fps, loss_fps

def get_name_parameters(fp):
    fp_split = fp.stem.split(',')
    fp_dict = {
        'subset': fp_split[0],
        'calibration_name': fp_split[1],
        'run_num': fp_split[2]
    }
    return fp_dict

def uniquify(path):
    '''
    Appends a number to the end of the outputs so that it doesn't overwrite previous calibration runs
    '''
    
    counter = 1
    original_stem = ','.join(path.stem.split(',')[0:-1]) # remove the zero part
    extension = path.suffix

    while path.exists():
        path = path.parent / (original_stem + f",{str(counter)}" + extension)
        counter += 1

    return path