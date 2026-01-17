"""
Forensic Analysis Services
SPEC-FORENSIC-001: Audio forensic analysis services for voice evidence analysis
SPEC-PERFOPT-001: Performance optimization with GPU memory and thermal management
"""

from voice_man.services.forensic.audio_feature_service import AudioFeatureService
from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
from voice_man.services.forensic.ser_service import SERService
from voice_man.services.forensic.cross_validation_service import CrossValidationService
from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService

# SPEC-PERFOPT-001 Phase 2: GPU Memory and Thermal Management
from voice_man.services.forensic.memory_manager import (
    ForensicMemoryManager,
    DEFAULT_STAGE_ALLOCATIONS,
)
from voice_man.services.forensic.thermal_manager import (
    ThermalManager,
    THROTTLE_START_TEMP,
    THROTTLE_STOP_TEMP,
    CRITICAL_TEMP,
)

# SPEC-PERFOPT-001 Phase 3: Pipeline Orchestration
from voice_man.services.forensic.pipeline_orchestrator import (
    PipelineOrchestrator,
    MAX_QUEUE_SIZE,
    BACKPRESSURE_RESUME_SIZE,
)

# SPEC-GPUOPT-001 Phase 3: GPU-Optimized Pipeline Orchestration
from voice_man.services.forensic.gpu_pipeline_orchestrator import (
    GPUPipelineOrchestrator,
    CPUPipelineOrchestrator,
)

# Report generation modules
from voice_man.reports.html_generator import ForensicHTMLGenerator
from voice_man.reports.pdf_generator import ForensicPDFGenerator, SimplePDFGenerator

__all__ = [
    "AudioFeatureService",
    "StressAnalysisService",
    "SERService",
    "CrossValidationService",
    "CrimeLanguageAnalysisService",
    "ForensicScoringService",
    # SPEC-PERFOPT-001 Phase 2: GPU Memory and Thermal Management
    "ForensicMemoryManager",
    "DEFAULT_STAGE_ALLOCATIONS",
    "ThermalManager",
    "THROTTLE_START_TEMP",
    "THROTTLE_STOP_TEMP",
    "CRITICAL_TEMP",
    # SPEC-PERFOPT-001 Phase 3: Pipeline Orchestration
    "PipelineOrchestrator",
    "MAX_QUEUE_SIZE",
    "BACKPRESSURE_RESUME_SIZE",
    # SPEC-GPUOPT-001 Phase 3: GPU-Optimized Pipeline Orchestration
    "GPUPipelineOrchestrator",
    "CPUPipelineOrchestrator",
    # Report generation
    "ForensicHTMLGenerator",
    "ForensicPDFGenerator",
    "SimplePDFGenerator",
]
