def is_number(s):
    """
    Somehow, the most pythonic way to check string for float number; used for safe user input parsing
    src: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
