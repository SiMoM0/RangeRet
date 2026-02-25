# Code from https://github.com/nschi00/rangeview-rgb-lidar-fusion/blob/new-main/modules/network/RangePreprocessLidar.py
import torch
import random

def match_elements(n):
    list1 = list(range(n))
    finished = False
    while not finished:
        try:
            list2 = list1.copy()
            random.shuffle(list2)
            matches = {}
            not_allowed = set()
            for i in range(n):
                current_element = list1[i]
                for j in range(n):
                    if list2[j] != current_element and (current_element, list2[j]) not in not_allowed:
                        matches[current_element] = list2[j]
                        not_allowed.add((current_element, list2[j]))
                        del list2[j]
                        break
            finished = True
        except:
            finished = False

    return matches

class RangeAugmentation():
    def __init__(self, aug_prob=[0.9, 0.2, 0.9, 1.0]):
        self.aug_prob = aug_prob

    def __call__(self, data, label, mask) -> torch.Any:
        B, _, _, _ = data.shape
        if B < 2:
            return data, label

        out_scan = []
        out_label = []
        match_dict = match_elements(B)
        for i in range(B):
            j = match_dict[i]

            scan_a, scan_b = data[i].clone(), data[j]
            label_a, label_b = label[i].clone(), label[j]
            mask_a = mask[i].clone()

            if torch.rand(1) < self.aug_prob[0]:
                scan_a, label_a = self.RangeMix(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[1]:
                scan_a, label_a = self.RangeUnion(scan_a, label_a, mask_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[2]:
                scan_a, label_a = self.RangePaste(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[3]:
                scan_a, label_a = self.RangeShift(scan_a, label_a)

            out_scan.append(scan_a)
            out_label.append(label_a)

        out_scan = torch.stack(out_scan)
        out_label = torch.stack(out_label)

        return out_scan, out_label.long()

    def RangeMix(self, scan_a, label_a, scan_b, label_b, mix_strategies=[2, 3, 4, 5]):
        B, H, W = scan_a.shape
        h_mix = random.choice(mix_strategies)
        w_mix = random.choice(mix_strategies)
        h_step = int(H/h_mix)
        w_step = int(W/w_mix)
        h_index = list(range(0, H, h_step))
        w_index = list(range(0, W, w_step))
        for i in range(len(h_index)):
            for j in range(len(w_index)):
                if (i + j) % 2 == 1:
                    h_s = h_index[i]
                    h_e = h_index[i] + h_step if h_index[i] + h_step < H else -1
                    w_s = w_index[j]
                    w_e = w_index[j] + w_step if w_index[j] + w_step < W else -1
                    scan_a[:, h_s:h_e, w_s:w_e] = scan_b[:, h_s:h_e, w_s:w_e]
                    label_a[h_s:h_e, w_s:w_e] = label_b[h_s:h_e, w_s:w_e]
        return scan_a, label_a

    def RangeUnion(self, scan_a, label_a, mask, scan_b, label_b, k=0.5):
        void = mask == 0
        mask_temp = torch.rand(void.shape, device=void.device) <= k
        # only fill 50% of the void points
        void = void.logical_and(mask_temp)
        scan_a[:, void], label_a[void] = scan_b[:, void], label_b[void]
        return scan_a, label_a

    def RangePaste(self, scan_a, label_a, scan_b, label_b):
        tail_classes = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]   # semantickitti
        #tail_classes = [1, 2, 5, 7, 12, 13]    # pandaset
        for cls in tail_classes:
            pix = label_b == cls
            scan_a[:, pix] = scan_b[:, pix]
            label_a[pix] = label_b[pix]
        return scan_a, label_a

    def RangeShift(self, scan_a, label_a):
        B, H, W = scan_a.shape
        p = torch.randint(low=int(0.25*W), high=int(0.75*W), size=(1,))
        scan_a = torch.cat([scan_a[:, :, p:], scan_a[:, :, :p]], dim=2)
        label_a = torch.cat([label_a[:, p:], label_a[:, :p]], dim=1)
        return scan_a, label_a