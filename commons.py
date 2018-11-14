
"""
This file exists solely to have helper functions that don't fit into other files in terms of purpose.
"""

def time_to_str(t):
	"""
	This function takes time in seconds as a float and converts it into a human readable string
	:param t: time in seconds
	:return: pretty string representation of t
	"""
	if t > 3600:
		return "{0:d}h {1:d}m {2:.3f}s".format(int(t // 3600), int((t % 3600) // 60), t % 60)
	elif t > 60:
		return "{0:d}m {1:.3f}s".format(int(t // 60), t % 60)
	else:
		return "{0:.3f}s".format(t)
