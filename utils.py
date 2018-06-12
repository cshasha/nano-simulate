import numpy as np
import os.path, sys, time, warnings
from datetime import datetime, timedelta

def timeit(method):
	"""Returns time elapsed by a particular function."""
	def timed(*args, **kw):
		ts = time.time()   #start time
		result = method(*args, **kw)
		te = time.time()   #end time
		microsec = timedelta(microseconds=(te-ts)*1000*1000)   #elapsed time in microseconds
		d = datetime(1,1,1) + microsec   #time format
		print(method.__name__ +  " %d:%d:%d:%d.%d" % (d.day-1, d.hour, d.minute, d.second, d.microsecond/1000))
		return result
	return timed