from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class SupervisedLearning(TrainerX):
    """Baseline model for domain adaptation, which is
    trained using source data only.
    """

    def forward_backward(self, batch_x):
        input, label = self.parse_batch_train(batch_x)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x):
        input = batch_x["img"]
        label = batch_x["label"]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)
        return input, label

    def inspect_extractor_contributions(self):
        backbone = self.model.backbone
        
        if not hasattr(backbone, "backbone_list"):
            print("Backbone is not fused; no extractors to inspect.")
            return

        num_extractors = len(backbone.backbone_list)
        project_dim = backbone.projections[0].out_features  

        classifier = self.model.classifier           # linear layer
        W = classifier.weight.detach()               # shape: (C, total_dim)

        contributions = []
        start = 0

        for idx, extractor_cls in enumerate(backbone.backbone_list):
            end = start + project_dim
            W_block = W[:, start:end]                 # slice for this extractor

            contrib = W_block.abs().sum().item()      # L1 importance

            contributions.append((extractor_cls.__name__, contrib))
            start = end

        # normalize
        total = sum(c for _, c in contributions)
        contributions_pct = [(name, c, c / total * 100) for name, c in contributions]

        print("\n=== Extractor Contribution Analysis ===")
        for name, raw, pct in contributions_pct:
            print(f"{name:<25} | Raw: {raw:10.4f} | {pct:6.2f}%")

        return contributions_pct
