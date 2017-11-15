import numpy as np
from processing.labels import dense_to_one_hot, one_hot_to_dense


# precision=tp/(tp+fp)
# recall=tp/(tp+fn)  # class accuracy
# IoU=tp/(tp+fn+fp)
# mIoU=class average IoU
# accuracy = (tp + tn)/(tp + tn + fp + fn)

# tp - true positives
# fp - false positives
# fn - false negatives
# fp - false positives

# recall - udio stvarnih piksela klase koji su pogodjeni (presjek/skup stvarnih piksela)
# precision - udio piksela pozitivno klasificiranih u klasu 1
# accuracy -

def compute_IoU(predictions, trues, classes, for_every_class=False):
	"""
	Computes mean intersection over union. 
	
	If for some reason, we want to see
	IoU for every class, you can set param for_every_class on True, and function will
	return dictionary where keys are classes and items are IoU values for each class.
	
	:param predictions: numpy array (values that NN predicted)
	:param trues: numpy array (values that are true)
	:param classes: numpy array (classes for which we want to inspect)
	:return: float || dict{int : float}
	"""
	dict_of_trues = dict()  # keys: classes, values: number of true predicted elements
	result_dict = dict()  # keys: classes, values: IoUs

	# initializing dictionaries
	for clas in classes:
		dict_of_trues[clas] = 0  # trues
		result_dict[clas] = 0

	# going through both predictions and trues
	for (pre, tru) in zip(predictions, trues):
		if pre == tru:  # where predictions are equal to trues,
			dict_of_trues[tru] += 1  # increase dict_of_trues on this specific key by 1

	sum_of_IoUs = 0

	# iterating over keys of result_dict
	for key in result_dict.keys():
		result_dict[key] = dict_of_trues[key]/len(predictions)  # calculating IoU for each class
		sum_of_IoUs += result_dict[key]  # adding that IoU to sum of IoUs

	if for_every_class:  # returning dictionary of IoUs for each class
		return result_dict
	else:				 # returning mean IoU
		return sum_of_IoUs/len(classes)


if __name__ == '__main__':

	# testing compute_IoU function
	pre = np.array([0, 1, 1])
	tru = np.array([0, 0, 0])
	classes = np.array([0, 1])
	result_dict = compute_IoU(pre, tru, classes, for_every_class=True)
	for key, item in result_dict.items():
		print("For class {}, IoU value is: {}".format(key, item))

	print(compute_IoU(pre, tru, classes))
