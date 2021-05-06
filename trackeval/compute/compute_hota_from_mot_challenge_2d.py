# Standard Library import
from os import makedirs, path
from shutil import copyfile, rmtree
from textwrap import dedent
from typing import List, Tuple

# Local import
from trackeval.eval import Evaluator
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.metrics import HOTA

# Decorator 
def _data_remover(function):
    """
    Wrapper that removes data folder in any case.
    """
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            if path.exists("./data/"):
                rmtree("./data")
    return wrapper

# Support functions
def _compute() -> dict:
    """
    Compute the hota by trackeval framework with desired eval_config

    Returns:
    dict : trackeval result dictionnary
    """
    hota_score_dict = {}

    # Set up config files
    eval_config = _get_costum_eval_config()
    dataset_config = _get_costum_dataset_config()

    # Run code
    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = [HOTA()]
    hota_score_dict, _ = evaluator.evaluate(dataset_list, metrics_list)
    return hota_score_dict

def _get_costum_eval_config() -> dict:
    """
    Costum eval file for computing hota with minimal output.
    See trackeval/eval.py for more informations about config

    Returns:
    dict : config file
    """
    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,  # Raises exception and exits with error
        "RETURN_ON_ERROR": False,  # if not BREAK_ON_ERROR, then returns from function on error
        "LOG_ON_ERROR": "./error_log.txt",  # if not None, save any errors into a log file.
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,  # If False, summary files are not output for classes with no detections
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }

    return eval_config

def _get_costum_dataset_config() -> dict:
    """
    Costum dataset file for computing hota with minimal output.
    See trackeval/datasets/mot_challenge_2d_box.py for more informations about
    config

    Returns:
    dict : config file
    """
    dataset_config = {
        "GT_FOLDER": "./data/gt/",  # Location of GT data
        "TRACKERS_FOLDER": "./data/trackers/",  # Trackers location
        "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
        "CLASSES_TO_EVAL": ["pedestrian"],  # Valid: ['pedestrian']
        "BENCHMARK": "dataset",  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        "SPLIT_TO_EVAL": "train",  # Valid: 'train', 'test', 'all'
        "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
        "PRINT_CONFIG": False,  # Whether to print current config
        "DO_PREPROC": True,  # Whether to perform preprocessing (never done for MOT15)
        "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        "SEQMAP_FOLDER": None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
        "SEQMAP_FILE": None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
        "SEQ_INFO": None,  # If not None, directly specify sequences to eval and their number of timesteps
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",  # '{gt_folder}/{seq}/gt/gt.txt'
        "SKIP_SPLIT_FOL": False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
        # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
        # If True, then the middle 'benchmark-split' folder is skipped for both.
    }

    return dataset_config

# Main function
@_data_remover
def compute_hota_from_mot_challenge_2d(
    gt_path_list: List[str], sequence_path_list: List[str]
) -> List[dict]:
    """
    Compute HOTA from MOT Challenge 2D file format.

    Arguments:
    gt_path_list        : List[str] List of paths of multiple gt file to be 
    evaluated by HOTA 
    sequence_path_list  : List[str] List of paths of multiple tracker sequence
    results evaluated by HOTA
    """
    hota_score_dict: dict = {}
    # Check if nb of gt files = nb of tracker result file
    if len(gt_path_list) is not len(sequence_path_list):
        print("Not same number element in both lists")
        return hota_score_dict
    nb_file: int = len(gt_path_list)

    # Creation of required hiearchy
    makedirs("./data/gt/dataset-train")
    makedirs("./data/gt/seqmaps")
    makedirs("./data/trackers/dataset-train/tracker/data")

    # Seqmaps file initialisation
    with open("./data/gt/seqmaps/dataset-train.txt", "a+") as file:
        file.write("name\n")
    
    # Setting up gt / tracker results files
    for i in range(nb_file):
        gt_path: str = gt_path_list[i]
        sequence_path: str = sequence_path_list[i]
        # Check if files exist
        if not (path.exists(gt_path) and path.exists(sequence_path)):
            print("FILE PATHS DON'T EXIST !")
            print("gt_path: {}".format(gt_path))
            print("sequence_path: {}".format(sequence_path))
            return hota_score_dict

        
        # Check if same seqLength for gt and sequence tracker result
        path_list : Tuple[str,str ]= ( gt_path, sequence_path )
        seqLength : int = 0
        for j,path_it in enumerate(path_list): 
            # Get seqLength for editing seqinfo.ini
            with open(path_it, "r") as file:
                for last_line in file:
                    pass
            split_tab = last_line.split(",")
            # First path seqLength
            if j == 0:
                seqLength : int = split_tab[0]
            # Second path seqLength is different from first one
            elif seqLength != split_tab[0]:
                print("GT AND SEQUENCE TRACKER RESULT DON'T HAVE THE SAME LENGTH")
                print("gt seqLength: {}".format(seqLength))
                print("sequence seqLength: {}".format(split_tab[0]))
                return hota_score_dict


        # Create data dir and files
        makedirs("./data/gt/dataset-train/seq_{!s}/gt".format(i + 1))
        copyfile(
            gt_path, "./data/gt/dataset-train/seq_{!s}/gt/gt.txt".format(i + 1)
        )
        copyfile(
            sequence_path,
            "./data/trackers/dataset-train/tracker/data/seq_{!s}.txt".format(
                i + 1
            ),
        )
        with open("./data/gt/seqmaps/dataset-train.txt", "a+") as file:
            file.write("seq_{!s}\n".format(i + 1))
        with open(
            "./data/gt/dataset-train/seq_{}/seqinfo.ini".format(i + 1), "a+"
        ) as file:
            file.write(
                dedent(
                    """\
                [Sequence]
                name=seq_{!s}
                seqLength={!s}
                """.format(
                        i + 1, seqLength
                    )
                )
            )

    # Run HOTA on MOT Challenge file, like run_mot_challenge_scripts
    hota_score_dict = _compute()
    return hota_score_dict
