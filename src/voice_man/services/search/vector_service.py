"""
FAISS 벡터 검색 서비스

시맨틱 검색(의미 기반 검색)을 위한 FAISS 벡터 인덱스 제공
- 임베딩 생성 및 저장
- 유사 벡터 검색
- 배치 처리 지원
"""

from __future__ import annotations

import pickle
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


class VectorSearchService:
    """
    FAISS 기반 벡터 검색 서비스

    FEATURES:
    - 벡터 임베딩 저장소
    - 코사인 유사도 검색
    - 배치 검색 지원
    - 디스크 저장/로드

    EXAMPLE:
        ```python
        # 서비스 초기화 (임베딩 차원 768)
        service = VectorSearchService(embedding_dim=768)

        # 벡터 추가
        service.add_vector(
            vector_id="ulid123",
            vector=embedding,
            metadata={"text": "안녕하세요", "speaker": "SPEAKER_00"}
        )

        # 검색
        results = service.search(
            query_vector=query_embedding,
            k=10
        )
        ```
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat",
        metric: str = "cosine",
    ) -> None:
        """
        벡터 검색 서비스 초기화

        Args:
            embedding_dim: 임베딩 차원 (기본값: 768 - KoBERT)
            index_type: 인덱스 타입 ('flat', 'ivf', 'hnsw')
            metric: 거리 메트릭 ('cosine', 'l2')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric

        # 벡터 저장
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # FAISS 인덱스 (lazy import)
        self._index = None

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        벡터 정규화 (L2 정규화)

        Args:
            vector: 정규화할 벡터

        Returns:
            정규화된 벡터
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        벡터 추가

        Args:
            vector_id: 벡터 ID
            vector: 임베딩 벡터
            metadata: 메타데이터 (선택 사항)
        """
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.embedding_dim}, got {vector.shape[0]}"
            )

        # 코사인 유사도를 위해 정규화
        normalized_vector = self._normalize_vector(vector)

        self.vectors[vector_id] = normalized_vector
        self.metadata[vector_id] = metadata or {}

        # 인덱스 무효화 (다음 검색 시 재구축)
        self._index = None

    def add_vectors_batch(
        self,
        vector_ids: List[str],
        vectors: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        배치 벡터 추가

        Args:
            vector_ids: 벡터 ID 리스트
            vectors: 임베딩 벡터 리스트
            metadata_list: 메타데이터 리스트 (선택 사항)
        """
        if len(vector_ids) != len(vectors):
            raise ValueError("vector_ids and vectors must have the same length")

        if metadata_list is not None and len(metadata_list) != len(vectors):
            raise ValueError("metadata_list must have the same length as vectors")

        for i, vector_id in enumerate(vector_ids):
            metadata = metadata_list[i] if metadata_list else None
            self.add_vector(vector_id, vectors[i], metadata)

    def remove_vector(self, vector_id: str) -> bool:
        """
        벡터 삭제

        Args:
            vector_id: 삭제할 벡터 ID

        Returns:
            삭제 성공 여부
        """
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            del self.metadata[vector_id]
            self._index = None
            return True
        return False

    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        벡터 조회

        Args:
            vector_id: 조회할 벡터 ID

        Returns:
            벡터 또는 None
        """
        return self.vectors.get(vector_id)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색

        Args:
            query_vector: 쿼리 벡터
            k: 반환할 결과 수
            filter_ids: 필터링할 ID 리스트 (선택 사항)

        Returns:
            검색 결과 리스트
            ```python
            [
                {
                    "id": "ulid123",
                    "score": 0.95,
                    "metadata": {"text": "안녕하세요", "speaker": "SPEAKER_00"}
                }
            ]
            ```
        """
        if not self.vectors:
            return []

        # 쿼리 벡터 정규화
        query_vector = self._normalize_vector(query_vector)

        # 필터링
        candidate_ids = set(filter_ids) if filter_ids else set(self.vectors.keys())

        # 코사인 유사도 계산 (내적)
        results = []
        for vector_id in candidate_ids:
            if vector_id not in self.vectors:
                continue

            vector = self.vectors[vector_id]
            # 코사인 유사도 = 정규화된 벡터의 내적
            similarity = float(np.dot(query_vector, vector))

            results.append(
                {
                    "id": vector_id,
                    "score": similarity,
                    "metadata": self.metadata.get(vector_id, {}),
                }
            )

        # 유사도 내림차순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)

        # 상위 k개 반환
        return results[:k]

    def search_batch(
        self,
        query_vectors: List[np.ndarray],
        k: int = 10,
        filter_ids: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        배치 벡터 검색

        Args:
            query_vectors: 쿼리 벡터 리스트
            k: 반환할 결과 수
            filter_ids: 필터링할 ID 리스트 (선택 사항)

        Returns:
            검색 결과 리스트의 리스트
        """
        return [self.search(qv, k, filter_ids) for qv in query_vectors]

    def save(self, filepath: Path | str) -> None:
        """
        인덱스를 디스크에 저장

        Args:
            filepath: 저장할 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vectors": self.vectors,
            "metadata": self.metadata,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path | str) -> "VectorSearchService":
        """
        디스크에서 인덱스 로드

        Args:
            filepath: 로드할 파일 경로

        Returns:
            로드된 VectorSearchService 인스턴스
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        service = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"],
            metric=data["metric"],
        )
        service.vectors = data["vectors"]
        service.metadata = data["metadata"]

        return service

    def clear(self) -> None:
        """모든 벡터 삭제"""
        self.vectors.clear()
        self.metadata.clear()
        self._index = None

    def get_stats(self) -> Dict[str, Any]:
        """
        인덱스 통계 조회

        Returns:
            인덱스 통계 정보
        """
        return {
            "total_vectors": len(self.vectors),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
        }

    def __len__(self) -> int:
        """벡터 수 반환"""
        return len(self.vectors)

    def __contains__(self, vector_id: str) -> bool:
        """벡터 존재 여부 확인"""
        return vector_id in self.vectors
