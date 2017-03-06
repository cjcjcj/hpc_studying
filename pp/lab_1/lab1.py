#!/usr/bin/env python3.5
import threading
import random
import time
 
from philosopher import Philosopher


def main(phil_num=5, run_time=100):
    forks = list(threading.Lock() for i in range(phil_num))
 
    philosophers = [
        Philosopher(str(i), forks[i%phil_num], forks[(i+1)%phil_num])
        for i in range(phil_num)
    ]
 
    random.seed()
    Philosopher.running = True
    for p in philosophers:
        p.start()

    time.sleep(run_time)
    Philosopher.running = False
 

if __name__ == '__main__':
    main()
