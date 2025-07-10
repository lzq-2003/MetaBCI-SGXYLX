def smooth(p_label):

    pred_al = self.a * p_label + (1 - self.a) * self.pred_all[-1]

    return pred_al
