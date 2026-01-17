"""
Training Data Generator for Dark Triad Classification
SPEC-CRIME-CLASS-001 Enhancement: BERT + Random Forest Hybrid

Generates synthetic training data based on psychological research for Dark Triad traits.
"""

import random
from typing import List, Dict, Tuple
import numpy as np


class DarkTriadTrainingDataGenerator:
    """
    Generate synthetic training data for Dark Triad personality traits

    Based on psychological research and the Short Dark Triad (SD3) questionnaire.
    Creates realistic text samples with varying trait levels.

    Attributes:
        seed: Random seed for reproducibility
    """

    # Trait templates based on SD3 research
    NARCISSISM_TEMPLATES = [
        # High narcissism
        "내가 제일 잘해, 다들 날 인정해야 해",
        "내 실력은 증명됐어, 누구도 날 못 따라와",
        "내가 한 방식이 최고야, 다른 방법은 필요 없어",
        "나는 특별해, 다들 난 부러워할 거야",
        "내 성과는 인정받아 마땅해",
        "내가 한 건 대단해, 칭찬받아야 해",
        "내 능력은 최고야, 누구도 날 넘볼 수 없어",
        "내가 리더야, 다들 날 따라야 해",
        "내 매력은 확실해, 다들 날 좋아할 거야",
        "내 성공은 당연해, 난 최고니까",
        # Medium narcissism
        "내가 좀 하는 편이지",
        "내 실력도 꽤 괜찮아",
        "내가 참여하면 더 잘될 거야",
        "내 아이디어도 꽤 좋아",
        # Low narcissism
        "우리 같이 하자",
        "네 의견도 좋아",
        "함께 고민해봐",
        "너도 할 수 있어",
    ]

    MACHIAVELLIANISM_TEMPLATES = [
        # High machiavellianism
        "결과가 중요해, 수단은 방법이야",
        "이득이 되면 해야지, 손해면 안 해",
        "전략적으로 접근해야 성공해",
        "상대를 이용하는 것도 방법이야",
        "목적을 위해선 수단과 방법을 안 가려",
        "상황에 맞게 처세해야 해",
        "계산적으로 행동해야 이득이야",
        "효율적으로 해결하는 게 핵심이야",
        "내 이익이 최우선이야",
        "승부에 이기는 게 다야",
        # Medium machiavellianism
        "잘 생각하고 결정할게",
        "상황을 봐서 할게",
        "실리적으로 판단할게",
        # Low machiavellianism
        "정직하게 하는 게 좋아",
        "원칙을 지켜야 해",
        "도움이 필요하면 말해줘",
        "함께 나누면 더 좋아",
    ]

    PSYCHOPATHY_TEMPLATES = [
        # High psychopathy
        "상관안해, 내가 할 거야",
        "책임질 생각 없어, 그냥 해",
        "후회안해, 다시는 안 그래",
        "감정에 휘둘리지 않아",
        "양심의 가책은 없어",
        "미안해할 생각 없어",
        "다들 상관없어, 나만 잘되면 돼",
        "무섭지도 않아, 그냥 해",
        "공감할 필요 없어",
        "죄책감은 느껴지지 않아",
        # Medium psychopathy
        "별거 아니야, 괜찮아",
        "크게 문제될 것 없어",
        "생각해볼 필요는 없어",
        # Low psychopathy
        "미안해, 신경 써줄게",
        "도와줄게, 괜찮아",
        "걱정해, 힘들 거야",
        "상황을 이해해",
    ]

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize training data generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

    def generate_single_sample(
        self, narcissism_level: float, machiavellianism_level: float, psychopathy_level: float
    ) -> str:
        """
        Generate a single text sample with specified trait levels

        Args:
            narcissism_level: Narcissism score (0.0 to 1.0)
            machiavellianism_level: Machiavellianism score (0.0 to 1.0)
            psychopathy_level: Psychopathy score (0.0 to 1.0)

        Returns:
            Generated text sample
        """
        parts = []

        # Add narcissism components
        if narcissism_level > 0.7:
            parts.extend(random.sample(self.NARCISSISM_TEMPLATES[:10], 2))
        elif narcissism_level > 0.4:
            parts.append(random.choice(self.NARCISSISM_TEMPLATES[10:14]))
        elif narcissism_level > 0.2:
            parts.append(random.choice(self.NARCISSISM_TEMPLATES[14:]))

        # Add machiavellianism components
        if machiavellianism_level > 0.7:
            parts.extend(random.sample(self.MACHIAVELLIANISM_TEMPLATES[:10], 2))
        elif machiavellianism_level > 0.4:
            parts.append(random.choice(self.MACHIAVELLIANISM_TEMPLATES[10:13]))
        elif machiavellianism_level > 0.2:
            parts.append(random.choice(self.MACHIAVELLIANISM_TEMPLATES[13:]))

        # Add psychopathy components
        if psychopathy_level > 0.7:
            parts.extend(random.sample(self.PSYCHOPATHY_TEMPLATES[:10], 2))
        elif psychopathy_level > 0.4:
            parts.append(random.choice(self.PSYCHOPATHY_TEMPLATES[10:13]))
        elif psychopathy_level > 0.2:
            parts.append(random.choice(self.PSYCHOPATHY_TEMPLATES[13:]))

        # Combine and add noise
        if parts:
            text = ", ".join(parts)
        else:
            # Neutral baseline
            text = random.choice(
                [
                    "오늘 날씨가 좋네요",
                    "밥 먹었어요?",
                    "안녕하세요",
                    "잘 지내요?",
                    "만나서 반가워요",
                ]
            )

        # Add some variation
        if random.random() > 0.5:
            connectors = ["근데", "그리고", "아니면", "게다가"]
            text = f"{text}, {random.choice(connectors)} "

        return text

    def generate_dataset(
        self,
        n_samples: int = 1000,
        label_distribution: str = "balanced",
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Generate complete training dataset

        Args:
            n_samples: Number of samples to generate
            label_distribution: Distribution strategy
                - "balanced": Equal distribution of trait combinations
                - "uniform": Random uniform distribution
                - "skewed": More low-trait samples (realistic)

        Returns:
            Tuple of (texts, labels) where labels is dict of trait -> binary array
        """
        texts = []
        labels = {
            "narcissism": [],
            "machiavellianism": [],
            "psychopathy": [],
        }

        # Define trait level combinations based on distribution
        if label_distribution == "balanced":
            # Generate equal samples from each combination
            combinations = []
            for n in [0, 1]:
                for m in [0, 1]:
                    for p in [0, 1]:
                        combinations.append((n, m, p))

            samples_per_combo = n_samples // len(combinations)

            for n_label, m_label, p_label in combinations:
                for _ in range(samples_per_combo):
                    # Generate trait levels (higher if label is 1)
                    n_level = random.uniform(0.6, 1.0) if n_label else random.uniform(0.0, 0.4)
                    m_level = random.uniform(0.6, 1.0) if m_label else random.uniform(0.0, 0.4)
                    p_level = random.uniform(0.6, 1.0) if p_label else random.uniform(0.0, 0.4)

                    text = self.generate_single_sample(n_level, m_level, p_level)
                    texts.append(text)
                    labels["narcissism"].append(n_label)
                    labels["machiavellianism"].append(m_label)
                    labels["psychopathy"].append(p_label)

        elif label_distribution == "uniform":
            # Random uniform distribution
            for _ in range(n_samples):
                n_level = random.random()
                m_level = random.random()
                p_level = random.random()

                # Binary labels based on threshold
                n_label = 1 if n_level > 0.5 else 0
                m_label = 1 if m_level > 0.5 else 0
                p_label = 1 if p_level > 0.5 else 0

                text = self.generate_single_sample(n_level, m_level, p_level)
                texts.append(text)
                labels["narcissism"].append(n_label)
                labels["machiavellianism"].append(m_label)
                labels["psychopathy"].append(p_label)

        elif label_distribution == "skewed":
            # Realistic distribution: more low-trait samples
            for _ in range(n_samples):
                # Use beta distribution to skew towards lower values
                n_level = np.random.beta(2, 5)  # Skewed low
                m_level = np.random.beta(2, 5)
                p_level = np.random.beta(2, 5)

                # Binary labels based on threshold
                n_label = 1 if n_level > 0.5 else 0
                m_label = 1 if m_level > 0.5 else 0
                p_label = 1 if p_level > 0.5 else 0

                text = self.generate_single_sample(n_level, m_level, p_level)
                texts.append(text)
                labels["narcissism"].append(n_label)
                labels["machiavellianism"].append(m_label)
                labels["psychopathy"].append(p_label)

        # Convert to numpy arrays
        labels = {trait: np.array(label_array) for trait, label_array in labels.items()}

        # Shuffle dataset
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = {trait: label_array[indices] for trait, label_array in labels.items()}

        return texts, labels

    def generate_regression_dataset(
        self, n_samples: int = 1000
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Generate dataset with continuous trait scores (0.0 to 1.0)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (texts, labels) where labels are continuous scores
        """
        texts = []
        labels = {
            "narcissism": [],
            "machiavellianism": [],
            "psychopathy": [],
        }

        for _ in range(n_samples):
            # Generate random trait levels (skewed towards lower values)
            n_level = np.random.beta(2, 5)
            m_level = np.random.beta(2, 5)
            p_level = np.random.beta(2, 5)

            text = self.generate_single_sample(n_level, m_level, p_level)
            texts.append(text)
            labels["narcissism"].append(n_level)
            labels["machiavellianism"].append(m_level)
            labels["psychopathy"].append(p_level)

        # Convert to numpy arrays
        labels = {trait: np.array(label_array) for trait, label_array in labels.items()}

        return texts, labels


# Convenience functions for quick generation
def generate_training_data(
    n_samples: int = 1000,
    distribution: str = "balanced",
    seed: int = 42,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Generate training dataset with default settings

    Args:
        n_samples: Number of samples to generate
        distribution: Label distribution strategy
        seed: Random seed

    Returns:
        Tuple of (texts, labels)
    """
    generator = DarkTriadTrainingDataGenerator(seed=seed)
    return generator.generate_dataset(n_samples=n_samples, label_distribution=distribution)


def generate_regression_data(
    n_samples: int = 1000, seed: int = 42
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Generate regression dataset with continuous scores

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Tuple of (texts, labels) with continuous scores
    """
    generator = DarkTriadTrainingDataGenerator(seed=seed)
    return generator.generate_regression_dataset(n_samples=n_samples)
