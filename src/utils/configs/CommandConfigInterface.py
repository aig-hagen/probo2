import abc
class CommandConfigInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'check') and 
                callable(subclass.check) and
                hasattr(subclass, 'print') and 
                callable(subclass.print) and 
                hasattr(subclass, 'dump') and 
                callable(subclass.dump) and 
                hasattr(subclass, 'to_string') and 
                callable(subclass.to_string))

    @abc.abstractmethod
    def check(self):
        """Check if configs are valid"""
        raise NotImplementedError

    @abc.abstractmethod
    def print(self):
        """Print configs to console"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def dump(self,path):
        """Dumps config to a file"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_String(self):
        """Returns config as string"""
        raise NotImplementedError
