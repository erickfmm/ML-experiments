
def is_integer(s, base=10):
	try:
		_ = int(s, base)
		return True
	except ValueError:
		return False


def is_float(s):
	try:
		_ = float(s)
		return True
	except ValueError:
		return False