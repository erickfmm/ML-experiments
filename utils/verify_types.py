
def is_integer(s, base=10):
	try:
		val = int(s, base)
		return True
	except ValueError:
		return False