"""
    Preparers are used to convert dataset in the original (raw) format to the
    standard format defined in AbstractPreparer.prepare. Such a prepared dateset
    can be loaded by the Dataset class.
    An example is Iccv09Preparer. It converts the ICCV09 (Stanford background)
    dataset to the desired format.
"""
import sys; sys.path.append('.')
from data.preparers.abstract_preparer import AbstractPreparer
from data.preparers.iccv09_preparer import Iccv09Preparer