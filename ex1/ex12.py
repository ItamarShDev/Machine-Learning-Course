if __name__ == "__main__":   
    """
    Gets Two Lists From the User
    If Every Value in the Second List
    is the Power of the Equivalent in the First
    it Will Print a List with the Sum of Each Cell

    INPUT:
    A Number Each Time
    To Change List Enter -1 
    """
    print """insert two lists
    one number at a time
    to end list enter -1"""
    count = 0
    a_list = []  #main list
    b_list = []  #the comparison list
    while True:
        try:
            a = int(raw_input("> "))  #scan from user
        except:  #not an integer
            print "Not a Valid Value, Exiting.."
            break
        if a == -1:  #stop the list
            count += 1
            if count == 1:
                print "first list", a_list
            if count == 2:  #have two lists
                print "second list", b_list
                break
        else:
            if count == 0:  #put in the first
                a_list.append(a)
            if count == 1:  #put in the second
                b_list.append(a)
    p_list = []  #merge list
    bool = False
    for a, b in zip(a_list, b_list):
        if b == a*a:
            p_list.append(a+b)
        else:
            bool = True
            break
    if bool:
        print "Comparison Failed"
    else:
        print ""
        print "The Joined List", p_list
