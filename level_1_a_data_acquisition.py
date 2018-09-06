import pandas as pd
import sys


def read_csv(*args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    return df


# This function works really well for one single function on scheduler (provided i had sys.stdout.flush() to the end of each file). But if more than one functions are running at the same time (different threads) the stdout
# saved is all mixed and saved on the file of the last function; - trying now with logging module
def log_files(project_name, output_dir='logs/'):
    sys.stdout = open(output_dir + project_name + '.txt', 'a')
    sys.stderr = open(output_dir + project_name + '.txt', 'a')



