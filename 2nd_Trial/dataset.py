"""
MDTD: Multimodal Depression Temporal Detector
Dataset loading and preprocessing for DAIC-WOZ
"""

import os
import re
import csv
import torch
import whisper
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torchaudio

from config import DataConfig


@dataclass
class QuestionBlock:
    """Single question-response block from interview"""
    participant_id: int
    block_idx: int
    question: str
    response: str
    audio_path: Optional[str]
    start_time: float
    end_time: float
    word_count: int


@dataclass
class InterviewSession:
    """Full interview session with multiple question blocks"""
    participant_id: int
    phq8_binary: int
    phq8_score: int
    blocks: List[QuestionBlock]


class DAICWOZDataset(Dataset):
    """
    DAIC-WOZ Dataset for depression detection.

    Loads transcripts and audio, segments into question blocks,
    and prepares features for the MDTD model.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        tokenizer: Optional[RobertaTokenizer] = None,
    ):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer or RobertaTokenizer.from_pretrained("roberta-base")

        # Load labels
        self.labels = self._load_labels()

        # Load and process sessions
        self.sessions = self._load_sessions()

        # Compile target question patterns
        self.question_patterns = self._compile_question_patterns()

    def _load_labels(self) -> Dict[int, Dict]:
        """Load PHQ-8 labels from CSV"""
        labels = {}
        label_file = os.path.join(
            self.config.daic_woz_dir,
            f"{self.split}_split_Depression_AVEC2017.csv"
        )

        # Try alternate naming convention
        if not os.path.exists(label_file):
            label_file = os.path.join(
                self.config.daic_woz_dir,
                "Depression_AVEC2017.csv"
            )

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = int(row["Participant_ID"])
                    labels[pid] = {
                        "phq8_binary": int(row.get("PHQ8_Binary", row.get("PHQ_Binary", 0))),
                        "phq8_score": int(row.get("PHQ8_Score", row.get("PHQ_Score", 0))),
                    }

        return labels

    def _compile_question_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for target questions"""
        patterns = []
        for q in self.config.target_questions:
            # Create flexible pattern that matches variations
            pattern = q.replace(" ", r"\s+")
            pattern = r".*" + pattern + r".*"
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns

    def _parse_transcript(self, transcript_path: str) -> List[Dict]:
        """Parse DAIC-WOZ transcript file"""
        turns = []

        with open(transcript_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 4:
                    try:
                        turns.append({
                            "start_time": float(row[0]),
                            "end_time": float(row[1]),
                            "speaker": row[2].strip(),
                            "text": row[3].strip(),
                        })
                    except (ValueError, IndexError):
                        continue

        return turns

    def _segment_question_blocks(
        self,
        turns: List[Dict],
        participant_id: int,
    ) -> List[QuestionBlock]:
        """Segment transcript into question-response blocks"""
        blocks = []
        current_question = None
        current_responses = []
        current_start = 0
        block_idx = 0

        for turn in turns:
            speaker = turn["speaker"].lower()

            if "ellie" in speaker:
                # Check if this is a target question
                is_target = any(p.match(turn["text"]) for p in self.question_patterns)

                if is_target:
                    # Save previous block if exists
                    if current_question and current_responses:
                        response_text = " ".join(current_responses)
                        blocks.append(QuestionBlock(
                            participant_id=participant_id,
                            block_idx=block_idx,
                            question=current_question,
                            response=response_text,
                            audio_path=None,  # Set later
                            start_time=current_start,
                            end_time=turn["start_time"],
                            word_count=len(response_text.split()),
                        ))
                        block_idx += 1

                    # Start new block
                    current_question = turn["text"]
                    current_responses = []
                    current_start = turn["end_time"]

            elif "participant" in speaker:
                if current_question:
                    current_responses.append(turn["text"])

        # Don't forget last block
        if current_question and current_responses:
            response_text = " ".join(current_responses)
            blocks.append(QuestionBlock(
                participant_id=participant_id,
                block_idx=block_idx,
                question=current_question,
                response=response_text,
                audio_path=None,
                start_time=current_start,
                end_time=turns[-1]["end_time"] if turns else current_start,
                word_count=len(response_text.split()),
            ))

        return blocks

    def _load_sessions(self) -> List[InterviewSession]:
        """Load all interview sessions"""
        sessions = []

        # Find all participant directories
        for pid in self.labels.keys():
            transcript_path = os.path.join(
                self.config.daic_woz_dir,
                f"{pid}_P",
                f"{pid}_TRANSCRIPT.csv"
            )

            if not os.path.exists(transcript_path):
                continue

            # Parse transcript and segment
            turns = self._parse_transcript(transcript_path)
            blocks = self._segment_question_blocks(turns, pid)

            if blocks:
                # Set audio paths
                audio_dir = os.path.join(self.config.daic_woz_dir, f"{pid}_P")
                for block in blocks:
                    audio_file = os.path.join(audio_dir, f"{pid}_AUDIO.wav")
                    if os.path.exists(audio_file):
                        block.audio_path = audio_file

                sessions.append(InterviewSession(
                    participant_id=pid,
                    phq8_binary=self.labels[pid]["phq8_binary"],
                    phq8_score=self.labels[pid]["phq8_score"],
                    blocks=blocks[:self.config.max_blocks_per_session],
                ))

        return sessions

    def _load_audio_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        """Load audio segment and convert to mel spectrogram"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.config.sample_rate
                )
                waveform = resampler(waveform)

            # Extract segment
            start_sample = int(start_time * self.config.sample_rate)
            end_sample = int(end_time * self.config.sample_rate)
            segment = waveform[:, start_sample:end_sample]

            # Convert to mono
            if segment.shape[0] > 1:
                segment = segment.mean(dim=0, keepdim=True)

            # Limit length
            max_samples = self.config.max_audio_length * self.config.sample_rate
            if segment.shape[1] > max_samples:
                segment = segment[:, :max_samples]

            # Convert to mel spectrogram using Whisper's preprocessing
            # Pad to 30 seconds for Whisper
            segment = segment.squeeze(0).numpy()
            segment = whisper.pad_or_trim(segment)
            mel = whisper.log_mel_spectrogram(segment)

            return mel  # (80, 3000)

        except Exception as e:
            # Return zero mel spectrogram on error
            return torch.zeros(80, 3000)

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using RoBERTa tokenizer"""
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single session"""
        session = self.sessions[idx]
        num_blocks = len(session.blocks)
        max_blocks = self.config.max_blocks_per_session

        # Initialize tensors
        text_input_ids = torch.zeros(max_blocks, self.config.max_text_length, dtype=torch.long)
        text_attention_mask = torch.zeros(max_blocks, self.config.max_text_length, dtype=torch.long)
        speech_mels = torch.zeros(max_blocks, 80, 3000)
        block_mask = torch.zeros(max_blocks)
        engagement_values = torch.zeros(max_blocks)

        # Process each block
        for i, block in enumerate(session.blocks):
            if i >= max_blocks:
                break

            # Tokenize text
            text_encoded = self._tokenize_text(block.response)
            text_input_ids[i] = text_encoded["input_ids"]
            text_attention_mask[i] = text_encoded["attention_mask"]

            # Load audio
            if block.audio_path:
                speech_mels[i] = self._load_audio_segment(
                    block.audio_path, block.start_time, block.end_time
                )

            # Engagement proxy: normalized word count
            engagement_values[i] = min(block.word_count / 100.0, 1.0)

            block_mask[i] = 1.0

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "speech_mels": speech_mels,
            "block_mask": block_mask,
            "engagement_values": engagement_values,
            "label": torch.tensor([session.phq8_binary], dtype=torch.float32),
            "participant_id": session.participant_id,
            "phq8_score": session.phq8_score,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader"""
    return {
        "text_input_ids": torch.stack([b["text_input_ids"] for b in batch]),
        "text_attention_mask": torch.stack([b["text_attention_mask"] for b in batch]),
        "speech_mels": torch.stack([b["speech_mels"] for b in batch]),
        "block_mask": torch.stack([b["block_mask"] for b in batch]),
        "engagement_values": torch.stack([b["engagement_values"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "participant_ids": [b["participant_id"] for b in batch],
        "phq8_scores": [b["phq8_score"] for b in batch],
    }


def create_dataloaders(
    config: DataConfig,
    train_config,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test dataloaders"""

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = DAICWOZDataset(config, split="train", tokenizer=tokenizer)
    val_dataset = DAICWOZDataset(config, split="dev", tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for debugging, increase for production
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Test loader (if available)
    test_loader = None
    try:
        test_dataset = DAICWOZDataset(config, split="test", tokenizer=tokenizer)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )
    except Exception:
        pass

    return train_loader, val_loader, test_loader
