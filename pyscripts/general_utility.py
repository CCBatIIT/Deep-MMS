from datetime import datetime

printf = lambda x : print(f"{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}//{x}", flush=True)

def printv(verbose:bool, x:str):
    if verbose:
        printf(x)
        