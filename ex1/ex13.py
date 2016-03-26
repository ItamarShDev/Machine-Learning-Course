

def is_int(x):
    """
    Validate if the Input is Integer
    Returns True if Yes
    ARGS:
    x - the value
    """
    try:
        int(x)  #try to cast to int
    except:
        return False  #failed
    return True  #suceeded

if __name__ == "__main__":
    """
    The Function Gets List Input From the User
    Prints Back an Array of the Integers

    INPUT:
    An One Value at a Time
    Ended with \'stop\'
    """
    print """
    Enter a Series of Number of Each Kind
    To Stop, Enter \'stop\'"""
    l = []
    while True:
        a = raw_input("> ")  #get the input
        if a == 'stop':  #stop event
            break
        l.append(a)
    res = filter(is_int, l)  #generate list on the integers in the input
    print "You Have Enter The Integers:", res
