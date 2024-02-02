
class NotAttempted(Exception):
    pass

class Incorrect(Exception):
    pass

class UserlandExceptionIncorrect(Incorrect):
    """Raised when a user's solution is incorrect because it raised an (unexpected)
    exception when called.
    """
    def __init__(self, exception, args):
        self.wrapped_exception = exception
        self.msg  = ("When calling your function with arguments `{!r}`, Python"
                " raised the following exception... **`{}: {}`**").format(
                        args, exception.__class__.__name__, exception)
    def __str__(self):
        return self.msg

class Uncheckable(Exception):
    pass

class ExceptionRaised(Exception): # Modified by GBM
    def __bool__(self):
        return False
    pass

class AnsweredC(Exception): # Modified by GBM
    def __bool__(self):
        return True
    pass

class AnsweredI(Exception): # Modified by GBM
    def __bool__(self):
        return False
    pass