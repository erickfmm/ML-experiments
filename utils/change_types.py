__all__ = ["to_int", "to_float", "list_to_str"]

def to_int(number, default, base=10, throw_error=False):
    x = default
    try:
        x = int(number, base)
    except:
        if throw_error:
            raise ValueError("not int")
    return x

def to_float(number, default, throw_error=False):
    x = default
    try:
        x = float(number)
    except:
        if throw_error:
            raise ValueError("not float")
    return x


def list_to_str(array_var, separator):
	s = ""
	for i in array_var:
		s += str(i)+separator
	return s[:-1]