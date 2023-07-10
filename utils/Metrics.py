import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import torch
from sklearn.metrics import average_precision_score


class Cls_Accuracy():
    def __init__(self, ):
        self.total = 0
        self.correct = 0

        self.precision = [torch.zeros(6), 0]
        self.recall = [torch.zeros(6), 0]
        self.f1 = [torch.zeros(6), 0]


    def update(self, logit, label):
        logit = logit.sigmoid_()
        logit = (logit >= 0.5)
        all_correct = torch.all(logit == label.byte(), dim=1).float().sum().item()

        label = label.detach().cpu()
        logit = logit.detach().cpu()

        for i, avg in enumerate([None, 'samples']):
            self.precision[i] += precision_score(label, logit, average=avg, zero_division=0) * logit.size(0)
            self.recall[i] += recall_score(label, logit, average=avg, zero_division=0) * logit.size(0)
            self.f1[i] += f1_score(label, logit, average=avg, zero_division=0) * logit.size(0)
        
        self.total += logit.size(0)
        self.correct += all_correct

    def compute_avg_acc(self):
        acc = self.correct / self.total
        precision = [self.precision[0] / self.total, self.precision[1] / self.total]
        recall = [self.recall[0] / self.total, self.recall[1] / self.total]
        f1 = [self.f1[0] / self.total, self.f1[1] / self.total]
        return acc, precision, recall, f1
    

class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        #intersection_over_union = intersection / (union.astype(np.float32) + 1e-4)
        intersection_over_union = intersection / union.astype(np.float32)

        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes) # only get foreground regions 
        
        hist = np.bincount(
            self.num_classes*label_true[mask] + label_pred[mask], 
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        recall = np.nanmean(recall) # mean recall 

        acc_cls = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        precision = np.nanmean(precision) # mean precision 

        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)

        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return {
            "Pixel_Accuracy": acc,
            "Recall" : recall,
            "Precision" : precision,
            "True Positive" : TP,
            "True Negative" : TN,
            "False Positive" : FP,
            "Mean_Accuracy": acc_cls,
            "Frequency_Weighted_IoU": fwavacc,
            "Mean_IoU": mean_iu,
            "Class_IoU": cls_iu,
        }


def calc_average_precision(gt_label, probs):
    ndata, nattr = gt_label.shape

    ap_list = []
    for i in range(nattr):
        y_true = gt_label[:, i]
        y_score = probs[:, i]

        ap_list.append(average_precision_score(y_true, y_score))
    ap = np.array(ap_list)
    mAP = ap.mean()
    return mAP, ap

def get_map_metrics(gt_label, probs, average='macro', all_metrics=True):
    
    mAP, ap = calc_average_precision(gt_label, probs)
    if average != 'macro':
        gt_label = gt_label[:, 1]
        probs = probs[:, 1]

    if all_metrics:
        acc = accuracy_score(gt_label, probs)
        pr = precision_score(gt_label, probs, pos_label=1, average=average)
        recall = recall_score(gt_label, probs, pos_label=1, average=average)
        f1 = f1_score(gt_label, probs, pos_label=1, average=average)
        return mAP, ap, acc, pr, recall, f1
    else:
        return mAP, ap

# same as calc_average_precision
def get_mAp(gt_label: np.ndarray, probs: np.ndarray):
    ndata, nattr = gt_label.shape
    rg = np.arange(1, ndata + 1).astype(float)
    ap_list = []
    for k in range(nattr):
        # sort scores
        scores = probs[:, k]
        targets = gt_label[:, k]
        sorted_idx = np.argsort(scores)[::-1]  # Descending
        truth = targets[sorted_idx]

        tp = np.cumsum(truth).astype(float)
        # compute precision curve
        precision = tp / rg

        # compute average precision
        ap_list.append(precision[truth == 1].sum() / max(truth.sum(), 1))

    ap = np.array(ap_list)
    mAp = ap.mean()
    return mAp, ap





def prob2metric(gt_label: np.ndarray, probs: np.ndarray, th):
    eps = 1e-6
    ndata, nattr = gt_label.shape

    # ------------------ macro, micro ---------------
    # gt_label[gt_label == -1] = 0
    pred_label = probs > th
    gt_pos = gt_label.sum(0)
    pred_pos = pred_label.sum(0)
    tp = (gt_label * pred_label).sum(0)

    OP = tp.sum() / pred_pos.sum()
    OR = tp.sum() / gt_pos.sum()
    OF1 = (2 * OP * OR) / (OP + OR)

    pred_pos[pred_pos == 0] = 1

    CP_all = tp / pred_pos
    CR_all = tp / gt_pos

    CP_all_t = tp / pred_pos
    CP_all_t[CP_all_t == 0] = 1
    CR_all_t = tp / gt_pos
    CR_all_t[CR_all_t == 0] = 1
    CF1_all = (2 * CP_all * CR_all) / (CP_all_t + CR_all_t)

    CF1_mean = CF1_all.mean()

    CP = np.mean(tp / pred_pos)
    CR = np.mean(tp / gt_pos)
    CF1 = (2 * CP * CR) / (CP + CR)

    gt_neg = ndata - gt_pos
    tn = ((1 - gt_label) * (1 - pred_label)).sum(0)

    label_pos_recall = 1.0 * tp / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * tn / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    ma = label_ma.mean()

    return OP, OR, OF1, CP, CR, CF1, ma, CP_all, CR_all, CF1_all, CF1_mean


def get_multilabel_metrics(gt_label, prob_pred, th=0.5):

    result = EasyDict()


    mAP, ap = calc_average_precision(gt_label, prob_pred)
    op, orecall, of1, cp, cr, cf1, ma, cp_all, cr_all, cf1_all, CF1_mean = prob2metric(gt_label, prob_pred, th)
    result.map = mAP * 100.

    # to json serializable
    result.CP_all = list(cp_all.astype(np.float64))
    result.CR_all = list(cr_all.astype(np.float64))
    result.CF1_all = list(cf1_all.astype(np.float64))
    result.CF1_mean = CF1_mean

    # simplified way
    # mAP, ap = calc_average_precision(gt_label, probs)
    # pred_label = probs > 0.5
    # CP, CR, _, _ = precision_recall_fscore_support(gt_label, pred_label, average='macro')
    # CF1 = 2 * CP * CR / (CP + CR)
    # OP, OR, OF1, _ = precision_recall_fscore_support(gt_label, pred_label, average='micro')

    result.OP = op * 100.
    result.OR = orecall * 100.
    result.OF1 = of1 * 100.
    result.CP = cp * 100.
    result.CR = cr * 100.
    result.CF1 = cf1 * 100.

    return result