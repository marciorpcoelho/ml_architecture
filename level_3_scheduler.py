import schedule
import time
from level_2_optionals_baviera import main as optionals_baviera

schedule.every().day.at("19:00").do(optionals_baviera)


while True:
    schedule.run_pending()
    time.sleep(1)
