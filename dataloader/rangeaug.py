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
    def __init__(self, aug_prob=[0.9, 0.7, 0.9, 0.0, 0.9], dataset='kitti'):
        self.aug_prob = aug_prob
        self.dataset = dataset

        if dataset == 'kitti':
            self.tail_classes = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19] # semantickitti
        elif dataset == 'pandaset':
            self.tail_classes = [1, 2, 5, 7, 12, 13] # pandaset
        elif dataset == 'poss':
            self.tail_classes = [2, 4, 6, 7, 8, 10] # semanticPOSS
        else:
            raise ValueError(f'Unsupported dataset {dataset} for range augmentation')
        
        print(f'[INFO] Using new range augmentations for {dataset} with probabilities {aug_prob}')

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
                scan_a, label_a = self.RangePolar(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[1]:
                scan_a, label_a = self.RangeBeams(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[2]:
                scan_a, label_a = self.RangeCompletion(scan_a, label_a, mask_a)
            if torch.rand(1) < self.aug_prob[3]:
                scan_a, label_a = self.RangeFake(scan_a, label_a)
            if torch.rand(1) < self.aug_prob[4]:
                scan_a, label_a = self.RangeInstance(scan_a, label_a, scan_b, label_b)

            out_scan.append(scan_a)
            out_label.append(label_a)

        out_scan = torch.stack(out_scan)
        out_label = torch.stack(out_label)

        return out_scan, out_label.long()

    def RangeInstance(self, scan_a, label_a, scan_b, label_b):
        tail_classes = self.tail_classes
        for cls in tail_classes:
            pix = label_b == cls
            scan_a[:, pix] = scan_b[:, pix]
            label_a[pix] = label_b[pix]
        return scan_a, label_a

    def RangeFake(self, scan_a, label_a):
        # substitute labels of some instances with underrepresented classes
        # NOTE: this augmentation may be unstable, however we can get best results even without it, so we can set its probability to 0 if needed
        pairs = {1: [2, 3, 5, 8], # car -> bicycle, motorcycle, other-vehicle, motorcyclist
                9: [10, 11, 12], # road -> parking, sidewalk, other-ground
        }
        rand_front_class = random.choice(list(pairs.keys()))
        rand_tail_class = random.choice(pairs[rand_front_class])
        # print(f'Randomly substituting {rand_front_class} with {rand_tail_class}')
        label_a[label_a == rand_front_class] = rand_tail_class
        return scan_a, label_a

    def RangeCompletion(self, scan_a, label_a, mask):
        # shift scan to the bottom of just one row below
        rows = 1
        shifted_scan = torch.zeros_like(scan_a)
        shifted_label = torch.zeros_like(label_a)
        shifted_scan[:, rows:, :] = scan_a[:, :-rows, :]
        shifted_label[rows:, :] = label_a[:-rows, :]
        # unite the two scans
        void = mask == 0
        scan_a[:, void], label_a[void] = shifted_scan[:, void], shifted_label[void]
        return scan_a, label_a

    def RangePolar(self, scan_a, label_a, scan_b, label_b):
        # mix two sections of range images
        B, H, W = scan_a.shape
        p_start = random.randint(0, 0.5*W)  # row index
        p_end = random.randint(0.5*W, W)  # row index
        scan_a[:, :, p_start:p_end] = scan_b[:, :, p_start:p_end]
        label_a[:, p_start:p_end] = label_b[:, p_start:p_end]
        # instance mixing and rotation
        tail_classes = self.tail_classes
        # extract only scan and labels values form a where label is in tail classes
        class_mask = torch.isin(label_a, torch.tensor(tail_classes, device=label_a.device))
        masked_scan_a = torch.full_like(scan_a, -1.0)
        masked_scan_a[:, class_mask] = scan_a[:, class_mask]
        masked_label_a = torch.full_like(label_a, -1)
        masked_label_a[class_mask] = label_a[class_mask]
        # flip the extracted mask
        class_mask = torch.flip(class_mask, dims=[1])
        masked_scan_a = torch.flip(masked_scan_a, dims=[2])
        masked_label_a = torch.flip(masked_label_a, dims=[1])
        # copy these values to scan_a and label_a
        scan_a[:, class_mask] = masked_scan_a[:, class_mask]
        label_a[class_mask] = masked_label_a[class_mask]
        return scan_a, label_a

    def RangeBeams(self, scan_a, label_a, scan_b, label_b):
        # mix different beams of range images
        B, H, W = scan_a.shape
        h_mix = random.choice([2, 3, 4])
        h_step = int(H/h_mix)
        h_index = list(range(0, H, h_step))
        for i in range(len(h_index)):
            if i % 2 == 1:
                h_s = h_index[i]
                h_e = h_index[i] + h_step if h_index[i] + h_step < H else -1
                scan_a[:, h_s:h_e, :] = scan_b[:, h_s:h_e, :]
                label_a[h_s:h_e, :] = label_b[h_s:h_e, :]
        return scan_a, label_a