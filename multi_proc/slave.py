__author__ = 'T'

import mmap
import time

from multi_proc.jsonmap import ObjectMmap


def main():
    mm = ObjectMmap(-1, 1024*1024, access=mmap.ACCESS_READ, tagname='share_mmap')
    while True:
        time.sleep(2)
        print '*' * 30
        print mm.jsonread_follower()

if __name__ == '__main__':
    main()