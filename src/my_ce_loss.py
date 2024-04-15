import torch

class MyCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        scaled_logits = logits - logits.max(dim=1).values.view(-1, 1)

        # scaled_logits = logits

        diagonals = torch.gather(scaled_logits, 1, labels.view(-1, 1))

        loss = torch.sum(torch.log(torch.exp(diagonals) / (torch.exp(scaled_logits).sum(dim=1).view(-1, 1))))

        return -loss / logits.size(0)


class MyCEAlignmentLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # scaled_logits = logits - logits.max(dim=1).values.view(-1, 1) # this is NOT equivalent to just logits
        scaled_logits = logits
 
        diagonals = torch.gather(scaled_logits, 1, labels.view(-1, 1))

        # loss = torch.sum(torch.log(torch.exp(diagonals))) # original, not numerically stable

        loss = torch.sum(diagonals) # because log exp cancels out

        return -loss / logits.size(0)
    