"""
MDTD: Multimodal Depression Temporal Detector
Evaluation and interpretation module
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from config import Config, get_config
from model import DepressionDetector, ModelOutput
from dataset import DAICWOZDataset, collate_fn
from utils import compute_metrics, load_checkpoint


@dataclass
class BlockAnalysis:
    """Analysis of a single question block"""
    block_idx: int
    question: str
    response_preview: str
    disagreement: float
    is_high_risk: bool


@dataclass
class TemporalAnalysis:
    """Temporal pattern analysis"""
    engagement_trend: str  # "DECLINING", "STABLE", "INCREASING"
    disagreement_trend: str
    engagement_slope: float
    disagreement_slope: float
    interpretation: str


@dataclass
class InterpretationReport:
    """Full interpretation report for a session"""
    participant_id: int
    depression_probability: float
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    confidence: str  # "HIGH", "MEDIUM", "LOW"

    # Cross-modal analysis (Idea 1)
    avg_disagreement: float
    cross_modal_interpretation: str

    # Temporal analysis (Idea 2)
    temporal: TemporalAnalysis

    # Block-level details
    num_blocks: int
    high_risk_blocks: List[BlockAnalysis]

    # Ground truth (if available)
    phq8_binary: Optional[int] = None
    phq8_score: Optional[int] = None


class Evaluator:
    """Evaluation and interpretation for MDTD model"""

    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
    ):
        self.config = config
        self.device = torch.device(config.train.device)

        # Load model
        self.model = DepressionDetector(config.model).to(self.device)
        load_checkpoint(self.model, checkpoint_path)
        self.model.eval()

        # Thresholds
        self.depression_threshold = config.eval.depression_threshold
        self.high_disagreement_threshold = config.eval.high_disagreement_threshold

    def _get_trend_label(self, slope: float, threshold: float = 0.05) -> str:
        """Convert slope to trend label"""
        if slope < -threshold:
            return "DECLINING"
        elif slope > threshold:
            return "INCREASING"
        else:
            return "STABLE"

    def _get_risk_level(self, prob: float) -> str:
        """Get risk level from probability"""
        if prob >= 0.7:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_confidence(self, prob: float) -> str:
        """Get prediction confidence"""
        distance_from_boundary = abs(prob - 0.5)
        if distance_from_boundary >= 0.3:
            return "HIGH"
        elif distance_from_boundary >= 0.15:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_cross_modal_interpretation(
        self,
        avg_disagreement: float,
        risk_level: str,
    ) -> str:
        """Generate interpretation for cross-modal disagreement"""
        if avg_disagreement > self.high_disagreement_threshold:
            if risk_level == "HIGH":
                return (
                    "환자의 말하는 내용과 목소리 톤 사이에 높은 불일치가 관찰됩니다. "
                    "이는 감정적 은폐나 사회적 바람직성 편향을 시사할 수 있습니다. "
                    "예: '괜찮다'고 말하지만 목소리는 무기력함."
                )
            else:
                return (
                    "내용-음성 불일치가 다소 높으나, 전반적 위험 수준은 낮습니다. "
                    "추가 관찰이 권장됩니다."
                )
        else:
            if risk_level == "HIGH":
                return (
                    "내용과 음성의 불일치는 낮으나, 다른 지표에서 우울 위험이 감지됩니다. "
                    "부정적 내용을 부정적 톤으로 일관되게 표현하는 패턴일 수 있습니다."
                )
            else:
                return (
                    "말하는 내용과 목소리 톤이 일치하며, "
                    "전반적으로 낮은 우울 위험을 나타냅니다."
                )

    def _generate_temporal_interpretation(
        self,
        engage_slope: float,
        disagree_slope: float,
    ) -> str:
        """Generate interpretation for temporal patterns"""
        interpretations = []

        if engage_slope < -0.05:
            interpretations.append(
                "대화가 진행될수록 참여도가 감소하는 패턴이 관찰됩니다. "
                "이는 피로감이나 동기 저하를 시사할 수 있습니다."
            )
        elif engage_slope > 0.05:
            interpretations.append(
                "대화 후반부로 갈수록 참여도가 증가합니다. "
                "긍정적 신호입니다."
            )

        if disagree_slope > 0.05:
            interpretations.append(
                "시간이 지남에 따라 내용-음성 불일치가 증가합니다. "
                "대화 초반에는 노력하다가 점점 지치는 패턴일 수 있습니다."
            )
        elif disagree_slope < -0.05:
            interpretations.append(
                "대화 후반부로 갈수록 내용과 음성이 더 일치합니다. "
                "라포(rapport)가 형성되며 편안해지는 것으로 해석될 수 있습니다."
            )

        if not interpretations:
            interpretations.append("시간에 따른 뚜렷한 변화 패턴은 관찰되지 않습니다.")

        return " ".join(interpretations)

    @torch.no_grad()
    def interpret_session(
        self,
        batch: Dict[str, torch.Tensor],
        questions: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
    ) -> InterpretationReport:
        """
        Generate interpretation report for a single session.

        Args:
            batch: Preprocessed batch from dataset
            questions: List of question texts (optional, for report)
            responses: List of response texts (optional, for report)

        Returns:
            InterpretationReport
        """
        # Move to device
        text_input_ids = batch["text_input_ids"].unsqueeze(0).to(self.device)
        text_attention_mask = batch["text_attention_mask"].unsqueeze(0).to(self.device)
        speech_mels = batch["speech_mels"].unsqueeze(0).to(self.device)
        block_mask = batch["block_mask"].unsqueeze(0).to(self.device)
        engagement_values = batch["engagement_values"].unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            speech_mels=speech_mels,
            block_mask=block_mask,
            engagement_values=engagement_values,
        )

        # Extract values
        prob = outputs.prob.item()
        avg_disagree = outputs.mean_disagreement.item()
        disagree_slope = outputs.disagreement_slope.item()
        engage_slope = outputs.engagement_slope.item()
        disagreements = outputs.disagreements.squeeze(0).cpu().numpy()

        # Risk assessment
        risk_level = self._get_risk_level(prob)
        confidence = self._get_confidence(prob)

        # Temporal analysis
        temporal = TemporalAnalysis(
            engagement_trend=self._get_trend_label(engage_slope),
            disagreement_trend=self._get_trend_label(disagree_slope),
            engagement_slope=engage_slope,
            disagreement_slope=disagree_slope,
            interpretation=self._generate_temporal_interpretation(
                engage_slope, disagree_slope
            ),
        )

        # Block-level analysis
        num_blocks = int(block_mask.sum().item())
        high_risk_blocks = []

        for i in range(num_blocks):
            is_high_risk = disagreements[i] > self.high_disagreement_threshold

            block_analysis = BlockAnalysis(
                block_idx=i,
                question=questions[i] if questions else f"Question {i+1}",
                response_preview=(
                    responses[i][:100] + "..." if responses and len(responses[i]) > 100
                    else responses[i] if responses else "N/A"
                ),
                disagreement=float(disagreements[i]),
                is_high_risk=is_high_risk,
            )

            if is_high_risk:
                high_risk_blocks.append(block_analysis)

        # Generate report
        report = InterpretationReport(
            participant_id=batch.get("participant_id", 0),
            depression_probability=prob,
            risk_level=risk_level,
            confidence=confidence,
            avg_disagreement=avg_disagree,
            cross_modal_interpretation=self._generate_cross_modal_interpretation(
                avg_disagree, risk_level
            ),
            temporal=temporal,
            num_blocks=num_blocks,
            high_risk_blocks=high_risk_blocks,
            phq8_binary=batch.get("label", torch.tensor([0])).item() if "label" in batch else None,
            phq8_score=batch.get("phq8_score", None),
        )

        return report

    def evaluate_dataset(
        self,
        dataset: DAICWOZDataset,
        output_dir: str,
    ) -> Dict[str, float]:
        """
        Evaluate on full dataset and generate reports.

        Args:
            dataset: DAICWOZDataset instance
            output_dir: Directory to save reports

        Returns:
            Evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        all_probs = []
        all_labels = []
        all_reports = []

        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            batch = dataset[idx]

            # Get session info
            session = dataset.sessions[idx]
            questions = [b.question for b in session.blocks]
            responses = [b.response for b in session.blocks]

            # Generate report
            report = self.interpret_session(batch, questions, responses)
            all_reports.append(report)

            all_probs.append(report.depression_probability)
            all_labels.append(report.phq8_binary or 0)

        # Compute metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        metrics = compute_metrics(all_probs, all_labels, self.depression_threshold)

        # Save reports
        reports_data = []
        for report in all_reports:
            report_dict = asdict(report)
            report_dict["temporal"] = asdict(report.temporal)
            report_dict["high_risk_blocks"] = [asdict(b) for b in report.high_risk_blocks]
            reports_data.append(report_dict)

        with open(os.path.join(output_dir, "interpretation_reports.json"), "w", encoding="utf-8") as f:
            json.dump(reports_data, f, indent=2, ensure_ascii=False)

        # Save metrics
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate summary
        self._generate_summary(all_reports, metrics, output_dir)

        return metrics

    def _generate_summary(
        self,
        reports: List[InterpretationReport],
        metrics: Dict[str, float],
        output_dir: str,
    ):
        """Generate human-readable summary"""
        summary_lines = [
            "# MDTD Evaluation Summary",
            "",
            "## Overall Metrics",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1 Score: {metrics['f1']:.4f}",
            f"- AUC-ROC: {metrics.get('auc_roc', 0):.4f}",
            "",
            "## Cross-Modal Disagreement Analysis",
        ]

        # Analyze disagreement by group
        dep_disagrees = [r.avg_disagreement for r in reports if r.phq8_binary == 1]
        ctrl_disagrees = [r.avg_disagreement for r in reports if r.phq8_binary == 0]

        if dep_disagrees:
            summary_lines.append(f"- Depression group mean disagreement: {np.mean(dep_disagrees):.4f}")
        if ctrl_disagrees:
            summary_lines.append(f"- Control group mean disagreement: {np.mean(ctrl_disagrees):.4f}")

        if dep_disagrees and ctrl_disagrees:
            diff = np.mean(dep_disagrees) - np.mean(ctrl_disagrees)
            summary_lines.append(f"- Difference: {diff:+.4f}")

        summary_lines.extend([
            "",
            "## Temporal Pattern Analysis",
        ])

        # Analyze temporal patterns by group
        dep_reports = [r for r in reports if r.phq8_binary == 1]
        ctrl_reports = [r for r in reports if r.phq8_binary == 0]

        if dep_reports:
            declining_pct = sum(1 for r in dep_reports if r.temporal.engagement_trend == "DECLINING") / len(dep_reports)
            summary_lines.append(f"- Depression group with declining engagement: {declining_pct:.1%}")

        if ctrl_reports:
            declining_pct = sum(1 for r in ctrl_reports if r.temporal.engagement_trend == "DECLINING") / len(ctrl_reports)
            summary_lines.append(f"- Control group with declining engagement: {declining_pct:.1%}")

        summary_lines.extend([
            "",
            "## Risk Level Distribution",
            f"- HIGH risk: {sum(1 for r in reports if r.risk_level == 'HIGH')}",
            f"- MEDIUM risk: {sum(1 for r in reports if r.risk_level == 'MEDIUM')}",
            f"- LOW risk: {sum(1 for r in reports if r.risk_level == 'LOW')}",
        ])

        with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))


def main():
    """Main entry point for evaluation"""
    import argparse
    from torch.utils.data import DataLoader
    from transformers import RobertaTokenizer

    parser = argparse.ArgumentParser(description="Evaluate MDTD model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    args = parser.parse_args()

    # Load config
    config = get_config()

    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)

    # Load dataset
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = DAICWOZDataset(config.data, split=args.split, tokenizer=tokenizer)

    # Evaluate
    metrics = evaluator.evaluate_dataset(dataset, args.output)

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output}")
    print(f"\nMetrics: {metrics}")


if __name__ == "__main__":
    main()
