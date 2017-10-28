class AbstractTrainingVisitor:
    def __init__(self):
        self.log = []
    
    def minibatch_completed(self, images, labels) -> bool:
        """
            Returns `True` if training needs to be interrupted.
        """
        return False
    
    def epoch_completed(self, images, labels) -> bool:
        """
            Returns `True` if training needs to be interrupted.
        """
        return False
    
    def add_log_message(self, message):
        self.log.append(message)
        

class DummyTrainingVisitor(AbstractTrainingVisitor):
    def __init__(self):
        pass