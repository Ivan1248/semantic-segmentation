import numpy as np
from processing.labels import dense_to_one_hot, one_hot_to_dense


# precision=tp/(tp+fp)
# recall=tp/(tp+fn)  # class accuracy
# IoU=tp/(tp+fn+fp)
# mIoU=class average IoU

def compute_precision_recall_iou_pa(predictions, trues):
    """
    pred, true: one_hot vector pixels (w,h,c)
    """
    n, shape = len(trues), trues[0].shape
    class_count = predictions[0].shape[2]
    tp = np.zeros(class_count)
    fp = np.zeros(class_count)
    fn = np.zeros(class_count)
    n = 0
    for pred, true in zip(predictions, trues):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for c in range(class_count):
                    if pred[i, j, c] == 1:
                        if true[i, j, c] == 1:
                            tp[c] += 1
                        else:
                            fp[c] += 1
                    elif true[i, j, c] == 1:
                        fn[c] += 1
        n += 1
        print(str(n) + '/' + str(len(trues)))

    precision = sum(0 if tp[i] == 0 else tp[i] / (tp[i] + fp[i]) for i in range(class_count)) / class_count
    recall = sum(0 if tp[i] == 0 else tp[i] / (tp[i] + fn[i]) for i in range(class_count)) / class_count
    iou = sum(0 if tp[i] == 0 else tp[i] / (tp[i] + fn[i] + fp[i]) for i in range(class_count)) / class_count
    pixel = sum(tp) / (n * shape[0] * shape[1])
    print("Precision: " + str(precision))
    print("Recall (class accuracy): " + str(recall))
    print("IOU accuracy: " + str(iou))
    print("Pixel accuracy: " + str(pixel))
    return precision, recall, iou, pixel


def class_accuracy(trues: list, predictions: list, class_count: int):
    results = np.zeros((class_count, 2))
    for l in range(len(trues)):
        # t = denso
        for i in range(trues[l].shape[0]):
            for j in range(trues[l].shape[1]):
                t = trues[l][i, j]
                results[t, t == predictions[l][i, j]] += 1
    return results


if __name__ == '__main__':
    pre = np.array([[[1, 0],
                     [0, 1],
                     [0, 1]]])

    tru = np.array([[[1, 0],
                     [1, 0],
                     [1, 0]]])

    a, b, c = compute_precision_recall_iou_pa([pre], [tru])
    pass
