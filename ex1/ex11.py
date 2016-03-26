if __name__ == "__main__":
    """
    Creates a Stack and Manages it by User Input.
    Options:
        'i' - Insert a Value - the value saved is from the 2nd index
        'e' - Delete Last
        'p' - Print the Stack Formattes to <[index] value>
    """
    print """Welcome!
    This Application Operates Stack For You
    press \'i\'' to insert a value
    press \'e\' to delete the last
    press\'p\' to print the stack
    the app will get out once the stack is empty"""
    stack = []
    while True:
        op = raw_input("> ")  #get input
        if op is 'i':  #insert a value
            str = raw_input(">> ")  #get value to insert
            if len(str) > 2:  #check the input is long enough
                stack.append(str[1:])  #insert to list
        elif op is 'e':  #pop last one
            if len(stack) == 0:  #get out if stack is empty
                break
            else:
                stack.pop()
        elif op is 'p':  #print the stack
            for i, a in enumerate(stack):
                print '[', i, ']', a
        else:
            print "Not a Valid Operation!\n"
