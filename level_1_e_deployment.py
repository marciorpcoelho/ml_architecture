import os


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    for i, df in enumerate(dfs):
        name = names[i] + '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)
