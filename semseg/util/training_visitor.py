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


class DummyTrainingVisitor(AbstractTrainingVisitor):
    def __init__(self):
        pass