import abc
class CommandOptionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'check') and 
                callable(subclass.check) and
                hasattr(subclass, 'print') and 
                callable(subclass.print))

    @abc.abstractmethod
    def check(self):
        """Check if options are valid"""
        raise NotImplementedError

    @abc.abstractmethod
    def print(self):
        """Print options to console"""
        raise NotImplementedError
