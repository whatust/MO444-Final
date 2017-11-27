# input.py
# define input functions for reading wiki database

def append_data(page, dat, m, days, x, y, num_days = 1):
    ### Appends to x, y the input/output
    ### input format: [page, current_day, access_of_previous_days]
    # page: a exclusive number given by the map m
    # current_day: a integer for the given month and days
    # access_of_previous_days: array of access of *num_days* previous days
    ### output: correct_number_of_access
    for i in range(num_days + 1, len(dat)):
        x += [[m[page]]]
        x[-1] += [date_to_value(days[i])]
        for j in range(1, num_days + 1):
            x[-1] += [dat[i - j]]
        y += [dat[i]]
    return x, y

def day_to_value(month, day):
    ## Given a given month and day, returns a fixed integer
    return month * 31 + day

def date_to_value(st):
    ## Given a string YYYY-MM-DD, returns a fixed integer (ignores the year)
    st = st.split('-')
    return day_to_value(int(st[1]), int(st[2]))

def treat(s):
    ## convert a line of input to an int array
    r = [""]
    f = False
    for i in range(0, len(s)):
        if(s[i] == "\""):
            f = not f
        if(f):
            r[-1] += s[i]
        else:
            if(s[i] == ','):
                r += [""]
            else:
                r[-1] += s[i]
    r = [i.strip() for i in r]
    for i in range(1, len(r)):
        if(r[i] == ""):
            r[i] = 0
        else:
            r[i] = int(float(r[i]))
    return r

