import os


def save_csv(df, name):
    # Checks for file existence and deletes it if exists, then saves it

    if os.path.isfile(name):
        os.remove(name)
    df.to_csv(name)