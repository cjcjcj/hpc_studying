import threading
import random
import time
from datetime import datetime

import settings


class Philosopher(threading.Thread):
 
    running = True
 
    def __init__(self, name, left_fork, right_fork):
        threading.Thread.__init__(self)
        self.name = name
        self.left_fork, self.right_fork = left_fork, right_fork
 
    def run(self):
        while(self.running):
            sleep_time = random.uniform(*settings.THINK_TIME)
            print(
                '[{}] {} goes to think for {:.4f} s.'.format(
                    datetime.now(), self.name, sleep_time
                )
            )
            time.sleep(sleep_time)
            print('[{}] {} is hungry.'.format(datetime.now(), self.name))
            self.dine()
 
    def dine(self):
        fork1, fork2 = self.left_fork, self.right_fork
 
        while self.running:
            fork1.acquire(True)
            locked = fork2.acquire(False)
            if locked:
                break
            fork1.release()
            print('[{}] {} swaps forks'.format(datetime.now(), self.name))
            fork1, fork2 = fork2, fork1
        else:
            return
 
        self.dining()
        fork2.release()
        fork1.release()
 
    def dining(self):           
        print('[{}] {} starts eating.'.format(datetime.now(), self.name))
        time.sleep(random.uniform(*settings.EAT_TIME))
        print(
            '[{}] {} finishes eating and leaves to think.'.format(
                datetime.now(), self.name
            )
        )
