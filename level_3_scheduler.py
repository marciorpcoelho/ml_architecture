import schedule
import time
import threading
import logging
import sys
from testing import main as test
from testing import main as test2
from level_2_optionals_baviera import main as optionals_baviera

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename='logs/optionals_baviera.txt', filemode='a')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()


schedule.every().day.at("18:03").do(run_threaded, optionals_baviera)
# schedule.every().day.at("17:47").do(run_threaded, test2)

while 1:
    schedule.run_pending()
    time.sleep(1)
