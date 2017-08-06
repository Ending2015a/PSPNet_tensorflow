import time
from datetime import datetime

import sys

class LogWriter:
    def __init__(self, path='pyLog.log'):
        self.filepath = path
        self.isOpened = False

    def open(self):
        try:
            if sys.version_info.major == 3:
                self.file = open(self.filepath, "a", encoding='UTF-8')
            else:
                self.file = open(self.filepath, "a")
        except (OSError, IOError) as e:
            print(str(e))

    def _writeLog(self, lgtype='INFO', msg='\n', ptimestamp=True, ptype=True):
        if self.isOpened == False:
            self.open()
            self.isOpened = True
        try:
            timestamp = ''
            type = ''
            if ptimestamp:
                timestamp = datetime.now()

            if ptype:
                type = ' ::' + lgtype + ':: '

            msg = '{0}{1}{2}\n'.format(timestamp, type, msg)
            self.file.write(msg)
            self.file.flush()
            if lgtype == 'ERROR':
                print(msg)
        except (OSError, IOError) as e:
            if lgtype == 'ERROR':
                print(msg)
            print(str(e))

    def clean(self):
        self.close()
        try:
            self.file = open(self.filepath, "w", encoding='UTF-8')
        except (OSError, IOError) as e:
            print(str(e))
    
    def Log(self, msg):
        self._writeLog('INFO', msg)


    def Error(self, msg):
        self._writeLog('ERROR', msg)

    def Write(self, msg, ptimestamp=False):
        self._writeLog(msg=msg, ptimestamp=ptimestamp, ptype=False)

    def setLog(self, path):
        self.file.close()
        self.filepath = path

    def close(self):
        if self.isOpened:
            self.file.close()
        self.isOpened = False

    def __del__(self):
        self.close()
