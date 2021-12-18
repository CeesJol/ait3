DEBUG=False
#DEBUG=True

def debug_q_update(prev_obs,  action, observation, reward, done, predict, gamma, future_val, target, td, new_q ):
    #print("Transition: %s" % ( ( prev_obs,  action, observation, reward, done), ) )
    if not DEBUG:
        return
        
    print("""Q update:
        predict = %f
        reward =  %f
        gamma =   %f
        futval =  %f
        target =  %f
        td =      %f
        new_q =   %f""" % (predict, reward, gamma, future_val, target, td, new_q) )

    if reward != 0 or target != 0:
        input("press enter to continue")

def callersname():
    import sys
    return sys._getframe(2).f_code.co_name

def nyi_warn(obj):
    s = "'%s()' not yet implemented for: '%s'" % (callersname(), obj.__str__())
    print(s)

def nyi_exc(obj):
    s = "'%s()' not yet implemented for: '%s'" % (callersname(), obj.__str__())
    raise Exception(s)

def override_exc(obj):
    s = "'%s()' not overriden by self: '%s'" % (callersname(), obj.__str__())
    raise Exception(s)


def assert_isinstance(v, cl):
    if not isinstance(v, cl):
        s =  "'%s' is no instance of %s!" %(v.__str__(), cl)
        raise Exception(s)

