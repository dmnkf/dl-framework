import torch
import torchmetrics
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_label_scenarios(device="cpu"):
    num_labels = 20

    # Test scenarios specifically for 20-label case
    test_cases = [
        {
            "name": "All labels correct",
            "preds": torch.full((100, 20), 0.7),
            "target": torch.ones((100, 20)),
            "expected": 1.0,
        },
        {
            "name": "All labels wrong",
            "preds": torch.full((100, 20), 0.7),
            "target": torch.zeros((100, 20)),
            "expected": 0.0,
        },
        {
            "name": "Exactly 5 labels correct per sample",
            "preds": torch.cat(
                [
                    torch.full((100, 5), 0.9),  # 5 correct predictions
                    torch.full((100, 15), 0.1),  # 15 never predicted positive
                ],
                dim=1,
            ),
            "target": torch.cat([torch.ones((100, 5)), torch.zeros((100, 15))], dim=1),
            # For macro, 5 labels have precision=1, 15 labels have precision=0 => avg=0.25
            "expected": 0.25,
        },
        {
            "name": "5 correct + 5 false positives",
            "preds": torch.cat(
                [
                    torch.full((100, 5), 0.9),  # 5 always correct
                    torch.full((100, 5), 0.9),  # 5 always false
                    torch.full((100, 10), 0.1),
                ],
                dim=1,
            ),
            "target": torch.cat(
                [
                    torch.ones((100, 5)),  # True positives for first 5 labels
                    torch.zeros((100, 15)),  # False (and predicted + negative)
                ],
                dim=1,
            ),
            # For macro, first 5 => precision=1, next 5 => precision=0, last 10 => 0 => avg=0.25
            "expected": 0.25,
        },
        {
            "name": "All False Predictions",
            "preds": torch.ones((1000, 20)),  # All predictions=1
            "target": torch.zeros((1000, 20)),  # All labels=0
            "expected": 0.0,
            "checks": ["no_nan", "exact_zero"],
        },
        {
            "name": "Numerical instability test",
            "preds": torch.randn(5000, 20).sigmoid(),
            "target": torch.randint(0, 2, (5000, 20)),
            "expected": None,
        },
    ]

    for case in test_cases:
        print(f"\n=== {case['name']} ===")

        preds = case["preds"].to(device)
        target = case["target"].to(device).long()

        precision = torchmetrics.Precision(
            task="multilabel", num_labels=num_labels, average="macro", threshold=0.5
        ).to(device)

        try:
            # Simulate batch updates
            for _ in range(10):
                precision.update(preds, target)

            result = precision.compute()
            logger.debug(f"Computed precision: {result}")

            if case["expected"] is not None:
                assert torch.isclose(
                    result, torch.tensor(case["expected"], device=device), atol=1e-7
                ), f"Expected {case['expected']}, got {result:.4f}"

            # Check for numerical stability
            assert not torch.isnan(result).any(), "NaN detected in precision"
            assert result.abs().max() < 1e6, f"Precision value exploded: {result}"

            print(f"✓ PASSED - Final Precision: {result:.4f}")
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
        finally:
            precision.reset()


if __name__ == "__main__":
    test_label_scenarios("mps")
