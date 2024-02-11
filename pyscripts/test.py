import sys

args = sys.argv

if len(args) == 1:
    print('arg not found')
    sys.exit(77)
else:
    print('arg found')
