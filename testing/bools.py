fns = ['2012_dfsdf', 'dsfsf', 'sdfsgf', 'sdfs', '2012_sdfsdf']

isShort = True

for fn in fns:
    if 'source' not in fn and (isShort or (not isShort and '2012' not in fn)):
        print(fn)
        
# isShort -> every fn
# NOT isShort -> only fns not containing 2012