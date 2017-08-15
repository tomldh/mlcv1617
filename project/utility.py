import logging

def logMsg(msg, use_log=False, printToConsole=True):
    
    if use_log:
        logging.debug(msg)
            
    if printToConsole:
        print(msg)