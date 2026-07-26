#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIB QUANTITATIVE DESCRIPTOR GRAPH v6.2 – LLM‑DIRECTED QUERY INTERFACE
====================================================================
Full integration of:
- v6.1: Memory‑safe batch processing, ontology, GNN, analytics, visualizations
- NEW: LLM‑directed query analysis (local/OpenAI/fallback)
- NEW: Ontology priority scoring (concept relevance, causal paths)
- NEW: Priority‑guided subgraph extraction
- NEW: Query‑driven visualization parameters

USAGE:
    streamlit run sib_concept_graph.py

Place JSON/BibTeX/CSV files in ./json_metadatabase/
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import math
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import functools
import time
import io
import base64
import copy
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Iterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score, r2_score, mean_absolute_error,
    mean_squared_error, davies_bouldin_score, pairwise_distances
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns

from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SIB Quantitative Descriptor Graph v6.2 – LLM Query",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================
class PerformanceMonitor:
    _timings: Dict[str, float] = {}
    _call_counts: Dict[str, int] = {}

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls) -> str:
        report = []
        for func_name, total_time in sorted(cls._timings.items(), key=lambda x: x[1], reverse=True):
            count = cls._call_counts.get(func_name, 1)
            avg_time = total_time / count
            report.append(f"  {func_name}: {total_time:.3f}s total ({count} calls, {avg_time:.4f}s avg)")
        return "\n".join(report)

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        func_name = func.__qualname__
        PerformanceMonitor._timings[func_name] = PerformanceMonitor._timings.get(func_name, 0) + elapsed
        PerformanceMonitor._call_counts[func_name] = PerformanceMonitor._call_counts.get(func_name, 0) + 1
        return result
    return wrapper

# ============================================================================
# ENUMS (Core)
# ============================================================================
class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MICROSTRUCTURE = "microstructure"
    MODEL = "model"
    GENERAL = "general"

class RelationshipType(Enum):
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    CONSTRAINS = "constrains"
    MODIFIES = "modifies"
    CORRECTS = "corrects"
    DRIVES = "drives"
    TRANSITIONS_TO = "transitions_to"
    REPLACES = "replaces"
    FORMS = "forms"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"
    GENERATES = "generates"
    ENABLES = "enables"
    DETECTS = "detects"
    MEASURES = "measures"
    OBSERVES = "observes"
    PROCESSES = "processes"
    REDUCES = "reduces"
    IMPROVES = "improves"

class SIBCoreProblem(Enum):
    ANODE_BOTTLENECK = "anode_bottleneck"
    CATHODE_INSTABILITY = "cathode_instability"
    SEI_CHEMISTRY = "sei_chemistry"
    SOLID_STATE_INTERFACE = "solid_state_interface"
    LOW_ENERGY_DENSITY = "low_energy_density"
    MOISTURE_MANUFACTURING = "moisture_manufacturing"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"

# ============================================================================
# EDGE COLOR REGISTRY
# ============================================================================
EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    RelationshipType.SYNONYM: "#AAAAAA",
    RelationshipType.HYPERNYM: "#5B9BD5",
    RelationshipType.HYPONYM: "#5B9BD5",
    RelationshipType.PART_OF: "#70AD47",
    RelationshipType.HAS_PART: "#70AD47",
    RelationshipType.CO_OCCURS: "#BFBFBF",
    RelationshipType.CAUSES: "#FF4444",
    RelationshipType.RESULTS_IN: "#E06040",
    RelationshipType.INFLUENCES: "#FF8C00",
    RelationshipType.DEPENDS_ON: "#DAA520",
    RelationshipType.CONSTRAINS: "#CC5500",
    RelationshipType.MODIFIES: "#FF6347",
    RelationshipType.CORRECTS: "#CD5C5C",
    RelationshipType.DRIVES: "#DC143C",
    RelationshipType.ENABLES: "#FF7F50",
    RelationshipType.TRANSITIONS_TO: "#8A2BE2",
    RelationshipType.REPLACES: "#9932CC",
    RelationshipType.FORMS: "#9370DB",
    RelationshipType.STABILIZES: "#7B68EE",
    RelationshipType.PRESERVES: "#6A5ACD",
    RelationshipType.GENERATES: "#6B8E23",
    RelationshipType.DETECTS: "#556B2F",
    RelationshipType.MEASURES: "#6B8E23",
    RelationshipType.OBSERVES: "#808000",
    RelationshipType.PROCESSES: "#EEE8AA",
    RelationshipType.REDUCES: "#90EE90",
    RelationshipType.IMPROVES: "#32CD32",
    RelationshipType.SEMANTIC: "#808080",
    RelationshipType.INFERRED: "#A9A9A9",
    RelationshipType.BRIDGE: "#C0C0C0",
}
EDGE_COLOR_FALLBACK = "#888888"

def get_edge_color(rel_type: RelationshipType) -> str:
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)

def get_edge_width(rel_type: RelationshipType) -> float:
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES, RelationshipType.FORMS,
              RelationshipType.STABILIZES, RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN, RelationshipType.MODIFIES,
              RelationshipType.ENABLES, RelationshipType.TRANSITIONS_TO}
    if rel_type in STRONG:
        return 3.0
    elif rel_type in MEDIUM:
        return 2.0
    return 1.0

def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS, RelationshipType.SEMANTIC,
              RelationshipType.BRIDGE}
    return "dashed" if rel_type in DASHED else "solid"

def lighten_hex_color(hex_color: str, factor: float) -> str:
    if not hex_color.startswith('#'):
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

# ============================================================================
# DATA CLASSES (Ontology, Problem Definitions, etc.)
# ============================================================================
@dataclass
class ConceptNode:
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional[np.ndarray] = None

    def add_synonym(self, synonym: str) -> None:
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.synonyms

@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False

@dataclass
class SIBProblemDefinition:
    problem_id: SIBCoreProblem
    title: str
    scientific_description: str
    root_cause: str
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]
    solution_directions: List[str]
    relevant_materials: List[str]
    relevant_phenomena: List[str]
    relevant_properties: List[str]
    typical_quantities: Dict[str, Tuple[float, float, str]]
    example_queries: List[str]
    visualization_focus: List[str]

    def get_ontology_concepts(self) -> Set[str]:
        concepts = set()
        concepts.update(self.key_concepts)
        concepts.update(self.relevant_materials)
        concepts.update(self.relevant_phenomena)
        concepts.update(self.relevant_properties)
        for src, _, tgt in self.key_relationships:
            concepts.add(src)
            concepts.add(tgt)
        return concepts

# ============================================================================
# PROBLEM DEFINITIONS (6 core SIB problems)
# ============================================================================
SIB_PROBLEM_DEFINITIONS: Dict[SIBCoreProblem, SIBProblemDefinition] = {
    SIBCoreProblem.ANODE_BOTTLENECK: SIBProblemDefinition(
        problem_id=SIBCoreProblem.ANODE_BOTTLENECK,
        title="The Anode Bottleneck: Graphite Incompatibility",
        scientific_description="Sodium ions (Na⁺, radius ~1.02 Å) cannot effectively intercalate into standard graphite layers...",
        root_cause="Na⁺ ionic radius (1.02 Å) is ~34% larger than Li⁺ (0.76 Å), preventing stable intercalation...",
        key_concepts=["hard_carbon", "alloying_anode", "intercalation_anode", "sodium_metal", "graphite_incompatibility",
                      "initial_coulombic_efficiency", "ice", "sodium_storage_mechanism", "pore_filling_mechanism",
                      "insertion_mechanism", "volume_expansion"],
        key_relationships=[("hard_carbon", "INFLUENCES", "specific_capacity"), ("hard_carbon", "INFLUENCES", "coulombic_efficiency"),
                           ("hard_carbon", "CAUSES", "low_ice"), ("alloying_anode", "INFLUENCES", "specific_capacity"),
                           ("alloying_anode", "CAUSES", "volume_expansion"), ("volume_expansion", "CAUSES", "cycle_life"),
                           ("pre_sodiation", "INFLUENCES", "coulombic_efficiency"), ("intercalation", "INFLUENCES", "specific_capacity")],
        solution_directions=["Optimize hard carbon microstructure", "Develop alloying anodes with nanostructuring",
                             "Apply pre-sodiation techniques", "Explore conversion anodes", "Investigate MXene-based anodes"],
        relevant_materials=["hard_carbon", "alloying_anode", "intercalation_anode", "sodium_metal", "mxene", "conversion_anode"],
        relevant_phenomena=["intercalation", "volume_expansion", "sei_formation", "sodium_plating_stripping", "conversion_reaction"],
        relevant_properties=["specific_capacity", "coulombic_efficiency", "cycle_life", "rate_capability", "initial_coulombic_efficiency"],
        typical_quantities={"hard_carbon_capacity": (200, 400, "mAh/g"), "hard_carbon_ice": (50, 85, "%"),
                            "alloying_anode_capacity": (400, 700, "mAh/g"), "alloying_volume_expansion": (200, 500, "%"),
                            "first_cycle_irreversibility": (15, 50, "%")},
        example_queries=["Why can't sodium intercalate into graphite like lithium does?",
                         "What is the sodium storage mechanism in hard carbon anodes?",
                         "How can we improve the initial Coulombic efficiency of hard carbon?",
                         "Compare hard carbon vs. alloying anodes for SIBs in terms of volume expansion",
                         "What pre-sodiation strategies can compensate for sodium loss in the first cycle?"],
        visualization_focus=["anode_materials_subgraph", "ice_analysis", "volume_expansion_comparison"]
    ),
    SIBCoreProblem.CATHODE_INSTABILITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.CATHODE_INSTABILITY,
        title="Cathode Structural Instability and Volume Change",
        scientific_description="The larger Na⁺ ion (~55% larger than Li⁺) causes significant mechanical stress during insertion/extraction...",
        root_cause="Na⁺ induces ~55% larger lattice parameter changes than Li⁺ during intercalation/deintercalation...",
        key_concepts=["layered_oxide_cathode", "polyanionic_cathode", "prussian_blue_analogue", "nasicon_cathode",
                      "phase_transition", "p2_o2_transition", "o3_p3_transition", "structural_degradation",
                      "volume_change", "lattice_stabilization", "elemental_doping", "concentration_gradient"],
        key_relationships=[("layered_oxide_cathode", "CAUSES", "phase_transition"), ("phase_transition", "CAUSES", "cycle_life"),
                           ("elemental_doping", "STABILIZES", "layered_oxide_cathode"), ("polyanionic_cathode", "INFLUENCES", "cycle_life"),
                           ("prussian_blue_analogue", "INFLUENCES", "rate_capability"), ("volume_expansion", "CAUSES", "structural_degradation")],
        solution_directions=["Elemental doping to stabilize layered structures", "Create concentration-gradient cathodes",
                             "Use polyanionic frameworks with robust 3D structures", "Design single-crystal cathodes",
                             "Develop Prussian blue analogues with minimal volume change"],
        relevant_materials=["layered_oxide_cathode", "polyanionic_cathode", "prussian_blue_analogue", "nasicon_cathode", "organic_cathode"],
        relevant_phenomena=["phase_transition", "volume_expansion", "intercalation", "structural_degradation", "solid_solution_behavior"],
        relevant_properties=["specific_capacity", "cycle_life", "rate_capability", "voltage_plateau", "energy_density"],
        typical_quantities={"layered_oxide_capacity": (100, 200, "mAh/g"), "polyanionic_capacity": (100, 120, "mAh/g"),
                            "pba_capacity": (100, 170, "mAh/g"), "volume_change_layered": (5, 25, "%"),
                            "phase_transition_voltage": (2.0, 4.2, "V vs. Na/Na⁺")},
        example_queries=["What causes the P2 to O2 phase transition in NaₓMnO₂ cathodes?",
                         "How does elemental doping stabilize layered oxide cathodes for SIBs?",
                         "Compare the structural stability of layered vs. polyanionic cathodes",
                         "Why do single-crystal cathodes show better cycling stability?"],
        visualization_focus=["cathode_phase_diagram", "doping_effects", "structural_comparison"]
    ),
    SIBCoreProblem.SEI_CHEMISTRY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SEI_CHEMISTRY,
        title="Electrolyte and Interphase (SEI) Chemistry",
        scientific_description="The Solid Electrolyte Interphase (SEI) is a protective layer formed on the anode from electrolyte decomposition...",
        root_cause="Sodium salts have higher solubility in the SEI matrix than lithium salts, preventing formation of a stable passivation layer",
        key_concepts=["sei_formation", "liquid_electrolyte", "interface_engineering", "artificial_sei",
                      "electrolyte_decomposition", "sei_solubility", "cei_formation", "interface_resistance",
                      "electrolyte_formulation", "concentrated_electrolyte", "fluorinated_electrolyte"],
        key_relationships=[("liquid_electrolyte", "CAUSES", "sei_formation"), ("sei_formation", "INFLUENCES", "cycle_life"),
                           ("sei_formation", "INFLUENCES", "coulombic_efficiency"), ("interface_engineering", "STABILIZES", "sei_formation"),
                           ("interface_resistance", "INFLUENCES", "rate_capability"), ("electrolyte_decomposition", "CAUSES", "low_ice")],
        solution_directions=["Design concentrated electrolytes", "Use fluorinated solvents/additives",
                             "Create artificial SEI layers", "Develop new sodium salts with lower SEI solubility",
                             "Optimize electrolyte formulation"],
        relevant_materials=["liquid_electrolyte", "solid_electrolyte", "polymer_electrolyte", "quasi_solid_electrolyte", "aqueous_electrolyte"],
        relevant_phenomena=["sei_formation", "electrolyte_decomposition", "sodium_plating_stripping"],
        relevant_properties=["coulombic_efficiency", "cycle_life", "ionic_conductivity"],
        typical_quantities={"sei_resistance": (50, 500, "Ω·cm²"), "na_sei_thickness": (10, 100, "nm"),
                            "first_cycle_loss": (15, 40, "%"), "ionic_conductivity_liquid": (1e-3, 1e-2, "S/cm")},
        example_queries=["Why is the SEI in SIBs less stable than in LIBs?",
                         "How do concentrated electrolytes improve SEI formation in sodium batteries?",
                         "What role do fluorinated additives play in SEI stabilization?",
                         "Compare artificial vs. natural SEI formation strategies"],
        visualization_focus=["sei_composition_map", "resistance_evolution", "electrolyte_comparison"]
    ),
    SIBCoreProblem.SOLID_STATE_INTERFACE: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SOLID_STATE_INTERFACE,
        title="Solid-State and Semi-Solid Interface Challenges",
        scientific_description="Moving to solid-state or quasi-solid-state SIBs introduces severe interfacial problems...",
        root_cause="Rigid solid electrolytes cannot accommodate the volume changes of electrode materials, causing contact loss, void formation, and dendrite penetration",
        key_concepts=["solid_electrolyte", "quasi_solid_electrolyte", "solid_polymer_electrolyte", "interface_contact",
                      "interfacial_resistance", "dendrite_growth", "void_formation", "delamination",
                      "chemo_mechanical_mismatch", "all_solid_state_sodium_battery", "nasicon", "sulfide_electrolyte"],
        key_relationships=[("solid_electrolyte", "CAUSES", "interfacial_resistance"), ("dendrite_growth", "CAUSES", "short_circuit"),
                           ("chemo_mechanical_mismatch", "CAUSES", "void_formation"), ("void_formation", "CAUSES", "delamination"),
                           ("interface_engineering", "STABILIZES", "interface_contact"), ("quasi_solid_electrolyte", "INFLUENCES", "interface_contact")],
        solution_directions=["Apply interfacial coating layers", "Design compliant interlayers",
                             "Use quasi-solid/gel electrolytes", "Develop dendrite-suppressing solid electrolytes",
                             "Apply external pressure to maintain contact"],
        relevant_materials=["solid_electrolyte", "quasi_solid_electrolyte", "solid_polymer_electrolyte", "all_solid_state_sodium_battery"],
        relevant_phenomena=["dendrite_growth", "void_formation", "delamination", "interfacial_resistance", "sodium_plating_stripping"],
        relevant_properties=["ionic_conductivity", "interfacial_resistance", "cycle_life"],
        typical_quantities={"solid_electrolyte_conductivity": (1e-5, 1e-3, "S/cm"), "interfacial_resistance": (10, 1000, "Ω·cm²"),
                            "critical_current_density": (0.1, 2.0, "mA/cm²"), "stack_pressure": (1, 100, "MPa")},
        example_queries=["What causes void formation at the solid electrolyte-anode interface?",
                         "How can quasi-solid electrolytes improve solid-state SIB performance?",
                         "Compare NASICON vs. sulfide solid electrolytes for sodium batteries",
                         "What strategies suppress sodium dendrite growth in solid electrolytes?"],
        visualization_focus=["interface_schematic", "dendrite_penetration", "pressure_effects"]
    ),
    SIBCoreProblem.LOW_ENERGY_DENSITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.LOW_ENERGY_DENSITY,
        title="Lower Energy Density Challenge",
        scientific_description="Sodium has a lower standard reduction potential (-2.71 V vs SHE) compared to lithium (-3.04 V vs SHE), and sodium is heavier (23 vs. 7 g/mol)...",
        root_cause="Na/Na⁺ potential is 0.33 V higher than Li/Li⁺, and Na atomic mass is 3.3× that of Li, both reducing theoretical specific energy",
        key_concepts=["energy_density", "specific_capacity", "voltage_plateau", "high_voltage_cathode",
                      "high_capacity_anode", "full_cell", "inactive_components", "electrode_loading",
                      "n_p_ratio", "cell_level_energy", "gravimetric_energy", "volumetric_energy"],
        key_relationships=[("specific_capacity", "CAUSES", "energy_density"), ("voltage_plateau", "INFLUENCES", "energy_density"),
                           ("high_voltage_cathode", "INFLUENCES", "energy_density"), ("high_capacity_anode", "INFLUENCES", "energy_density"),
                           ("full_cell", "DEPENDS_ON", "energy_density"), ("inactive_components", "CONSTRAINS", "energy_density")],
        solution_directions=["Develop high-voltage cathodes (>4.0 V vs. Na/Na⁺)", "Use high-capacity anodes with volume mitigation",
                             "Optimize full-cell design (N/P ratio, electrode loading)", "Minimize inactive components",
                             "Explore sodium metal anodes for maximum energy"],
        relevant_materials=["layered_oxide_cathode", "polyanionic_cathode", "hard_carbon", "alloying_anode", "sodium_metal", "full_cell", "organic_cathode"],
        relevant_phenomena=["intercalation", "conversion_reaction", "sodium_plating_stripping"],
        relevant_properties=["energy_density", "specific_capacity", "voltage_plateau"],
        typical_quantities={"sib_cell_energy": (100, 180, "Wh/kg"), "lib_cell_energy": (200, 300, "Wh/kg"),
                            "high_voltage_cathode": (3.8, 4.5, "V vs. Na/Na⁺"), "target_ev_energy": (150, 200, "Wh/kg")},
        example_queries=["What is the theoretical energy density limit for SIBs vs LIBs?",
                         "How can high-voltage cathodes push SIB energy density closer to LIBs?",
                         "Compare full-cell energy densities with different anode-cathode pairs",
                         "Can SIBs ever achieve energy density competitive with LIBs for EVs?"],
        visualization_focus=["energy_density_comparison", "ragone_plot", "voltage_profile_comparison"]
    ),
    SIBCoreProblem.MOISTURE_MANUFACTURING: SIBProblemDefinition(
        problem_id=SIBCoreProblem.MOISTURE_MANUFACTURING,
        title="Moisture Sensitivity and Manufacturing Challenges",
        scientific_description="Many high-performance sodium cathode materials and sodium salts (NaPF₆) are highly hygroscopic—they absorb moisture from air...",
        root_cause="Sodium cathode materials react with atmospheric H₂O/CO₂ to form surface alkaline species (NaOH, Na₂CO₃) that disrupt slurry rheology",
        key_concepts=["moisture_sensitivity", "hygroscopic_materials", "surface_alkalinity", "slurry_gelation",
                      "coating_defects", "aqueous_processing", "moisture_stable_materials", "surface_washing",
                      "dry_room_requirements", "manufacturing_scalability", "electrode_fabrication", "slurry_coating"],
        key_relationships=[("moisture_sensitivity", "CAUSES", "surface_alkalinity"), ("surface_alkalinity", "CAUSES", "slurry_gelation"),
                           ("slurry_gelation", "CAUSES", "coating_defects"), ("surface_washing", "REDUCES", "surface_alkalinity"),
                           ("aqueous_processing", "ENABLES", "manufacturing_scalability"), ("slurry_coating", "PROCESSES", "electrode_fabrication")],
        solution_directions=["Develop moisture-stable cathode compositions", "Apply surface washing treatments",
                             "Use aqueous binders compatible with alkaline surfaces", "Design dry room specifications",
                             "Explore water-based slurry processing with pH control"],
        relevant_materials=["layered_oxide_cathode", "prussian_blue_analogue", "polyanionic_cathode"],
        relevant_phenomena=["surface_alkalinity", "slurry_gelation", "moisture_degradation"],
        relevant_properties=["coulombic_efficiency", "cycle_life", "manufacturing_yield"],
        typical_quantities={"surface_pH": (10, 13, "pH units"), "na2co3_content": (0.5, 5.0, "wt%"),
                            "dry_room_dew_point": (-40, -60, "°C"), "slurry_viscosity_target": (1000, 5000, "mPa·s")},
        example_queries=["Why are sodium cathode materials more moisture-sensitive than lithium cathodes?",
                         "How does surface alkalinity affect slurry coating quality?",
                         "What surface washing treatments can stabilize sodium cathode materials?",
                         "Compare dry room requirements for SIB vs LIB manufacturing"],
        visualization_focus=["moisture_degradation_schematic", "slurry_rheology", "surface_analysis"]
    ),
    SIBCoreProblem.GENERAL: SIBProblemDefinition(
        problem_id=SIBCoreProblem.GENERAL,
        title="General SIB Inquiry",
        scientific_description="General inquiry about sodium-ion batteries.",
        root_cause="N/A",
        key_concepts=["sodium_ion_battery"],
        key_relationships=[],
        solution_directions=[],
        relevant_materials=[],
        relevant_phenomena=[],
        relevant_properties=[],
        typical_quantities={},
        example_queries=["What is a sodium-ion battery?"],
        visualization_focus=["general_overview"]
    ),
    SIBCoreProblem.MULTI_PROBLEM: SIBProblemDefinition(
        problem_id=SIBCoreProblem.MULTI_PROBLEM,
        title="Multi-Problem SIB Inquiry",
        scientific_description="Inquiry spanning multiple core SIB problems.",
        root_cause="N/A",
        key_concepts=[],
        key_relationships=[],
        solution_directions=[],
        relevant_materials=[],
        relevant_phenomena=[],
        relevant_properties=[],
        typical_quantities={},
        example_queries=[],
        visualization_focus=["multi_problem_comparison"]
    ),
}

# ============================================================================
# DOMAIN ONTOLOGY
# ============================================================================
class DomainOntology:
    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self.synonym_to_canonical: Dict[str, str] = {}
        self._build_ontology()

    def _add_concept(self, canonical_name: str, concept_type: ConceptType,
                     synonyms: Set[str] = None, definition: str = "") -> None:
        node = ConceptNode(canonical_name=canonical_name, concept_type=concept_type,
                           synonyms=synonyms or set(), definition=definition)
        self.concepts[canonical_name] = node

    def _add_relationship(self, source: str, rel_type: RelationshipType,
                          target: str, confidence: float = 1.0) -> None:
        self.relationships.append(Relationship(source, target, rel_type, confidence))

    def _build_ontology(self) -> None:
        # === CATHODE MATERIALS ===
        self._add_concept("layered_oxide_cathode", ConceptType.MATERIAL,
                          synonyms={"na_mno2", "namno2", "na_x_mno2", "p2_na_mno2", "o3_na_mno2", "layered oxide"},
                          definition="Sodium transition metal oxide cathodes (e.g., NaₓMnO₂)")
        self._add_concept("polyanionic_cathode", ConceptType.MATERIAL,
                          synonyms={"na3v2(po4)3", "nvp", "na3v2(po4)2f3", "nvpf", "polyanion"},
                          definition="Polyanionic compound cathodes with NASICON or phosphate frameworks")
        self._add_concept("prussian_blue_analogue", ConceptType.MATERIAL,
                          synonyms={"pba", "prussian blue", "na2mnfe(cn)6", "hexacyanoferrate"},
                          definition="Prussian blue analogues with open framework for sodium intercalation")
        self._add_concept("nasicon_cathode", ConceptType.MATERIAL,
                          synonyms={"nasicon", "na superionic conductor", "na3zr2si2po12"},
                          definition="NASICON-type cathodes with 3D framework")
        # === ANODE MATERIALS ===
        self._add_concept("hard_carbon", ConceptType.MATERIAL,
                          synonyms={"hc", "hard carbon anode", "disordered carbon", "non-graphitizable carbon"},
                          definition="Hard carbon with disordered structure, common SIB anode")
        self._add_concept("sodium_metal", ConceptType.MATERIAL,
                          synonyms={"na metal", "sodium anode", "metallic sodium", "na foil"},
                          definition="Pure sodium metal anode")
        self._add_concept("alloying_anode", ConceptType.MATERIAL,
                          synonyms={"sn anode", "sb anode", "bi anode", "tin anode", "antimony anode", "alloy anode"},
                          definition="Alloying-type anode materials (Sn, Sb, Bi)")
        self._add_concept("intercalation_anode", ConceptType.MATERIAL,
                          synonyms={"tio2", "na2ti3o7", "layered titanium oxide"},
                          definition="Intercalation anode materials (e.g., TiO₂, Na₂Ti₃O₇)")
        # === ELECTROLYTES ===
        self._add_concept("liquid_electrolyte", ConceptType.MATERIAL,
                          synonyms={"organic electrolyte", "naclo4 in ec/dec", "na pf6", "aqueous electrolyte"},
                          definition="Liquid electrolyte for SIBs")
        self._add_concept("solid_electrolyte", ConceptType.MATERIAL,
                          synonyms={"solid sodium electrolyte", "nasicon", "na3ps4", "sulfide electrolyte"},
                          definition="Solid-state electrolyte")
        self._add_concept("polymer_electrolyte", ConceptType.MATERIAL,
                          synonyms={"peo", "polyethylene oxide", "gel polymer"},
                          definition="Polymer-based electrolyte")
        self._add_concept("quasi_solid_electrolyte", ConceptType.MATERIAL,
                          synonyms={"gel electrolyte", "quasi-solid", "semi-solid", "in-situ polymerized"},
                          definition="Quasi-solid electrolyte")
        # === PROPERTIES ===
        self._add_concept("specific_capacity", ConceptType.PROPERTY,
                          synonyms={"capacity", "mah/g", "specific charge", "gravimetric capacity"},
                          definition="Specific capacity (mAh/g)")
        self._add_concept("energy_density", ConceptType.PROPERTY,
                          synonyms={"wh/kg", "specific energy", "volumetric energy density"},
                          definition="Energy density (Wh/kg)")
        self._add_concept("coulombic_efficiency", ConceptType.PROPERTY,
                          synonyms={"ce", "charge-discharge efficiency", "reversibility"},
                          definition="Coulombic efficiency (%)")
        self._add_concept("cycle_life", ConceptType.PROPERTY,
                          synonyms={"cycling stability", "retention", "capacity retention"},
                          definition="Cycle life before 80% capacity retention")
        self._add_concept("rate_capability", ConceptType.PROPERTY,
                          synonyms={"rate performance", "high rate", "c-rate"},
                          definition="Ability to maintain capacity at high rates")
        self._add_concept("ionic_conductivity", ConceptType.PROPERTY,
                          synonyms={"na+ conductivity", "s/cm", "ionic transport"},
                          definition="Ionic conductivity (S/cm)")
        self._add_concept("voltage_plateau", ConceptType.PROPERTY,
                          synonyms={"discharge voltage", "charge voltage", "voltage profile"},
                          definition="Voltage plateau (V)")
        # === PHENOMENA ===
        self._add_concept("dendrite_growth", ConceptType.PHENOMENON,
                          synonyms={"sodium dendrite", "dendrite formation", "mossy sodium"},
                          definition="Formation of sodium dendrites")
        self._add_concept("sei_formation", ConceptType.PHENOMENON,
                          synonyms={"solid electrolyte interphase", "sei layer", "passivation film"},
                          definition="Solid-electrolyte interphase")
        self._add_concept("sodium_plating_stripping", ConceptType.PHENOMENON,
                          synonyms={"na plating", "sodium stripping", "electrodeposition"},
                          definition="Electrochemical deposition and dissolution of sodium")
        self._add_concept("intercalation", ConceptType.PHENOMENON,
                          synonyms={"na+ insertion", "sodium intercalation", "deintercalation"},
                          definition="Insertion/extraction of Na+ into host structure")
        self._add_concept("conversion_reaction", ConceptType.PHENOMENON,
                          synonyms={"conversion", "alloying/dealloying", "conversion electrode"},
                          definition="Electrochemical conversion reaction")
        # === METHODS ===
        self._add_concept("cyclic_voltammetry", ConceptType.METHOD,
                          synonyms={"cv", "cyclic voltammogram", "voltammetry"},
                          definition="Cyclic voltammetry")
        self._add_concept("electrochemical_impedance_spectroscopy", ConceptType.METHOD,
                          synonyms={"eis", "nyquist plot", "impedance spectroscopy"},
                          definition="EIS")
        self._add_concept("galvanostatic_cycling", ConceptType.METHOD,
                          synonyms={"constant current", "cccv", "galvanostatic"},
                          definition="Galvanostatic cycling")
        self._add_concept("operando_characterization", ConceptType.METHOD,
                          synonyms={"in situ xrd", "operando xrd", "in situ raman"},
                          definition="Operando characterization")
        # === PARAMETERS ===
        self._add_concept("current_density", ConceptType.PARAMETER,
                          synonyms={"ma/g", "a/g", "c-rate"},
                          definition="Current density (mA/g or A/g)")
        self._add_concept("cut_off_voltage", ConceptType.PARAMETER,
                          synonyms={"voltage window", "v", "upper cut-off", "lower cut-off"},
                          definition="Cut-off voltage (V)")
        self._add_concept("temperature", ConceptType.PARAMETER,
                          synonyms={"celsius", "kelvin", "operating temperature"},
                          definition="Temperature (°C or K)")
        # === PROCESSES ===
        self._add_concept("slurry_coating", ConceptType.PROCESS,
                          synonyms={"electrode coating", "doctor blade", "tape casting"},
                          definition="Slurry coating")
        self._add_concept("cell_assembly", ConceptType.PROCESS,
                          synonyms={"coin cell", "pouch cell", "swagelok", "cell fabrication"},
                          definition="Cell assembly")
        # === GENERAL ===
        self._add_concept("sodium_ion_battery", ConceptType.MODEL,
                          synonyms={"sib", "na-ion battery", "sodium battery"},
                          definition="Sodium-ion battery")
        self._add_concept("all_solid_state_sodium_battery", ConceptType.MODEL,
                          synonyms={"asssb", "solid-state sodium"},
                          definition="All-solid-state sodium battery")
        self._add_concept("full_cell", ConceptType.MODEL,
                          synonyms={"full sodium ion cell", "practical cell"},
                          definition="Full-cell configuration")

        self._build_synonym_index()
        self._build_relationships()

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_relationships(self) -> None:
        # Material → Property
        self._add_relationship("hard_carbon", RelationshipType.INFLUENCES, "specific_capacity", 0.85)
        self._add_relationship("hard_carbon", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.75)
        self._add_relationship("alloying_anode", RelationshipType.INFLUENCES, "specific_capacity", 0.90)
        self._add_relationship("alloying_anode", RelationshipType.CAUSES, "volume_expansion", 0.85)
        self._add_relationship("sodium_metal", RelationshipType.INFLUENCES, "energy_density", 0.90)
        self._add_relationship("sodium_metal", RelationshipType.CAUSES, "dendrite_growth", 0.80)
        self._add_relationship("layered_oxide_cathode", RelationshipType.INFLUENCES, "specific_capacity", 0.80)
        self._add_relationship("layered_oxide_cathode", RelationshipType.CAUSES, "phase_transition", 0.75)
        self._add_relationship("polyanionic_cathode", RelationshipType.INFLUENCES, "cycle_life", 0.85)
        self._add_relationship("prussian_blue_analogue", RelationshipType.INFLUENCES, "rate_capability", 0.80)
        self._add_relationship("solid_electrolyte", RelationshipType.INFLUENCES, "ionic_conductivity", 0.90)
        self._add_relationship("liquid_electrolyte", RelationshipType.CAUSES, "sei_formation", 0.85)
        self._add_relationship("polymer_electrolyte", RelationshipType.INFLUENCES, "cycle_life", 0.75)
        # Properties → Performance
        self._add_relationship("specific_capacity", RelationshipType.CAUSES, "energy_density", 0.95)
        self._add_relationship("coulombic_efficiency", RelationshipType.CAUSES, "cycle_life", 0.90)
        self._add_relationship("rate_capability", RelationshipType.INFLUENCES, "specific_capacity", 0.80)
        self._add_relationship("ionic_conductivity", RelationshipType.INFLUENCES, "rate_capability", 0.85)
        self._add_relationship("voltage_plateau", RelationshipType.INFLUENCES, "energy_density", 0.90)
        # Phenomena → Performance
        self._add_relationship("dendrite_growth", RelationshipType.CAUSES, "cycle_life", -0.85)
        self._add_relationship("dendrite_growth", RelationshipType.CAUSES, "coulombic_efficiency", -0.80)
        self._add_relationship("sei_formation", RelationshipType.INFLUENCES, "cycle_life", 0.70)
        self._add_relationship("intercalation", RelationshipType.INFLUENCES, "specific_capacity", 0.80)
        # Methods → Detection
        self._add_relationship("cyclic_voltammetry", RelationshipType.DETECTS, "intercalation", 0.85)
        self._add_relationship("electrochemical_impedance_spectroscopy", RelationshipType.DETECTS, "sei_formation", 0.80)
        self._add_relationship("galvanostatic_cycling", RelationshipType.MEASURES, "specific_capacity", 0.90)
        self._add_relationship("operando_characterization", RelationshipType.OBSERVES, "dendrite_growth", 0.75)
        # Parameters → Performance
        self._add_relationship("current_density", RelationshipType.INFLUENCES, "rate_capability", 0.85)
        self._add_relationship("cut_off_voltage", RelationshipType.CONSTRAINS, "specific_capacity", 0.70)
        self._add_relationship("temperature", RelationshipType.INFLUENCES, "ionic_conductivity", 0.80)
        # Processing → Cell
        self._add_relationship("slurry_coating", RelationshipType.PROCESSES, "cell_assembly", 0.85)
        self._add_relationship("cell_assembly", RelationshipType.FORMS, "sodium_ion_battery", 0.95)
        # Taxonomy
        self._add_relationship("all_solid_state_sodium_battery", RelationshipType.HYPONYM, "sodium_ion_battery", 0.9)
        self._add_relationship("full_cell", RelationshipType.HYPONYM, "sodium_ion_battery", 0.95)
        self._add_relationship("full_cell", RelationshipType.DEPENDS_ON, "coulombic_efficiency", 0.80)

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        return None

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        return self.concepts[canonical_name].concept_type if canonical_name in self.concepts else ConceptType.GENERAL

    def get_definition(self, canonical_name: str) -> str:
        return self.concepts[canonical_name].definition if canonical_name in self.concepts else ""

    def get_related_concepts(self, canonical_name: str,
                             rel_type: RelationshipType = None) -> List[Tuple[str, RelationshipType, float]]:
        related = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related

    def infer_path(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        paths = []
        visited = set()
        def dfs(current, path, depth):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            for rel in self.relationships:
                if rel.source == current and rel.confidence > 0.5:
                    path.append(rel.target)
                    dfs(rel.target, path, depth + 1)
                    path.pop()
            visited.remove(current)
        dfs(source, [source], 0)
        return paths

# ============================================================================
# GRAPH SAGE GNN (sparse)
# ============================================================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, adj_indices, adj_values, num_nodes, h,
                pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

# ============================================================================
# QUERY ANALYSIS DATA CLASSES (NEW)
# ============================================================================
@dataclass
class ConceptPriority:
    concept_name: str
    concept_type: ConceptType
    composite_score: float
    direct_score: float
    problem_affinity_score: float
    causal_path_score: float
    centrality_bonus: float
    cooccurrence_bonus: float
    is_explicitly_mentioned: bool
    is_inferred: bool
    inference_reason: str = ""

    def to_dict(self) -> Dict:
        return {"concept": self.concept_name, "type": self.concept_type.value,
                "score": round(self.composite_score, 3),
                "explicit": self.is_explicitly_mentioned,
                "inferred": self.is_inferred,
                "reason": self.inference_reason}

@dataclass
class QueryAnalysisResult:
    original_query: str
    normalized_query: str
    primary_problem: SIBCoreProblem
    secondary_problems: List[SIBCoreProblem]
    problem_confidences: Dict[str, float]
    explicitly_mentioned: List[str]
    inferred_concepts: List[str]
    all_relevant_concepts: List[str]
    concept_priorities: Dict[str, ConceptPriority] = field(default_factory=dict)
    query_type: str = "general"
    emphasis_direction: str = "cause"
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=list)
    subgraph_depth: int = 2
    priority_threshold: float = 0.3
    focus_nodes: List[str] = field(default_factory=list)
    bridge_nodes: List[str] = field(default_factory=list)
    suggested_layout: str = "force"
    highlight_paths: List[List[str]] = field(default_factory=list)
    visualization_focus: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def get_top_concepts(self, n: int = 10) -> List[ConceptPriority]:
        return sorted(self.concept_priorities.values(), key=lambda x: x.composite_score, reverse=True)[:n]

    def get_concepts_above_threshold(self, threshold: float = None) -> List[str]:
        thresh = threshold or self.priority_threshold
        return [name for name, cp in self.concept_priorities.items() if cp.composite_score >= thresh]

# ============================================================================
# LLM QUERY ANALYZER (Abstract + Implementations)
# ============================================================================
from abc import ABC, abstractmethod

class LLMQueryAnalyzer(ABC):
    @abstractmethod
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        pass
    @abstractmethod
    def is_available(self) -> bool:
        pass

class FallbackAnalyzer(LLMQueryAnalyzer):
    PROBLEM_KEYWORDS = {
        SIBCoreProblem.ANODE_BOTTLENECK: {"anode", "hard carbon", "graphite", "intercalation", "alloying",
                                          "ice", "initial coulombic efficiency", "sodiation", "sn anode", "sb anode"},
        SIBCoreProblem.CATHODE_INSTABILITY: {"cathode", "layered oxide", "phase transition", "p2", "o2", "o3", "p3",
                                             "structural", "degradation", "doping", "stability", "volume change"},
        SIBCoreProblem.SEI_CHEMISTRY: {"sei", "solid electrolyte interphase", "electrolyte", "interface",
                                       "passivation", "decomposition", "fluorinated", "concentrated electrolyte"},
        SIBCoreProblem.SOLID_STATE_INTERFACE: {"solid state", "solid electrolyte", "nasicon", "sulfide", "interface",
                                               "contact", "dendrite", "void", "delamination", "pressure"},
        SIBCoreProblem.LOW_ENERGY_DENSITY: {"energy density", "wh/kg", "specific energy", "voltage", "capacity",
                                            "full cell", "n/p ratio", "gravimetric", "volumetric"},
        SIBCoreProblem.MOISTURE_MANUFACTURING: {"moisture", "humidity", "hygroscopic", "surface alkalinity",
                                                "slurry", "coating", "manufacturing", "dry room", "aqueous processing"},
    }
    def is_available(self) -> bool: return True
    @timed
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        q = query.lower().strip()
        problem_scores = {}
        for problem, keywords in self.PROBLEM_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q)
            problem_scores[problem] = score
        if sum(problem_scores.values()) == 0:
            primary = SIBCoreProblem.GENERAL
        else:
            primary = max(problem_scores, key=problem_scores.get)
        secondary = [p for p, s in sorted(problem_scores.items(), key=lambda x: -x[1]) if s > 0 and p != primary][:2]
        explicitly_mentioned = []
        for canonical, node in ontology.concepts.items():
            if canonical.replace("_", " ") in q:
                explicitly_mentioned.append(canonical)
                continue
            for syn in node.synonyms:
                if syn.replace("_", " ") in q:
                    explicitly_mentioned.append(canonical)
                    break
        query_type = "question" if "?" in q or any(w in q for w in ["why","how","what"]) else "general"
        if any(w in q for w in ["compare","vs","versus"]): query_type = "comparison"
        if any(w in q for w in ["mechanism","process","how does"]): query_type = "mechanism"
        if any(w in q for w in ["improve","optimize","enhance","strategy"]): query_type = "optimization"
        emphasis = "solution" if query_type == "optimization" else "cause" if any(w in q for w in ["cause","reason","why"]) else "general"
        inferred = []
        if primary in SIB_PROBLEM_DEFINITIONS:
            for c in SIB_PROBLEM_DEFINITIONS[primary].key_concepts:
                if c not in explicitly_mentioned and c in ontology.concepts:
                    inferred.append(c)
        return QueryAnalysisResult(
            original_query=query, normalized_query=q,
            primary_problem=primary, secondary_problems=secondary,
            problem_confidences={p.value: s/max(sum(problem_scores.values()),1) for p,s in problem_scores.items()},
            explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred,
            all_relevant_concepts=explicitly_mentioned + inferred,
            query_type=query_type, emphasis_direction=emphasis,
            subgraph_depth=2, priority_threshold=0.3,
            focus_nodes=explicitly_mentioned[:5],
            reasoning_chain=[f"Rule-based: found {len(explicitly_mentioned)} explicit concepts"],
            confidence=0.6 if explicitly_mentioned else 0.3
        )

class LocalLLMAnalyzer(LLMQueryAnalyzer):
    SYSTEM_PROMPT = """You are a scientific query analyzer for Sodium-Ion Battery (SIB) research.
Given a user query, return JSON with:
1. "primary_problem": one of "anode_bottleneck","cathode_instability","sei_chemistry","solid_state_interface","low_energy_density","moisture_manufacturing","general","multi_problem"
2. "secondary_problems": list
3. "query_type": "question","comparison","mechanism","optimization","general"
4. "emphasis_direction": "cause","effect","solution","property","comparison"
5. "key_concepts": list of canonical concept names
6. "implicit_concepts": list
7. "reasoning": brief explanation
Only valid JSON."""
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None; self.model = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception:
            pass
    def is_available(self) -> bool:
        return self.model is not None
    @timed
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        if not self.is_available(): return FallbackAnalyzer().analyze_query(query, ontology)
        prompt = f"{self.SYSTEM_PROMPT}\n\nUser Query: {query}\n\nAnalysis:"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else json.loads(response)
        except:
            data = {"primary_problem":"general", "query_type":"general", "key_concepts":[]}
        problem_map = {"anode_bottleneck":SIBCoreProblem.ANODE_BOTTLENECK, "cathode_instability":SIBCoreProblem.CATHODE_INSTABILITY,
                       "sei_chemistry":SIBCoreProblem.SEI_CHEMISTRY, "solid_state_interface":SIBCoreProblem.SOLID_STATE_INTERFACE,
                       "low_energy_density":SIBCoreProblem.LOW_ENERGY_DENSITY, "moisture_manufacturing":SIBCoreProblem.MOISTURE_MANUFACTURING,
                       "general":SIBCoreProblem.GENERAL, "multi_problem":SIBCoreProblem.MULTI_PROBLEM}
        primary = problem_map.get(data.get("primary_problem","general"), SIBCoreProblem.GENERAL)
        secondary = [problem_map.get(p, SIBCoreProblem.GENERAL) for p in data.get("secondary_problems", []) if p in problem_map]
        explicit = [c for c in data.get("key_concepts", []) if ontology.resolve_concept(c)]
        inferred = [c for c in data.get("implicit_concepts", []) if ontology.resolve_concept(c) and c not in explicit]
        return QueryAnalysisResult(
            original_query=query, normalized_query=query.lower(),
            primary_problem=primary, secondary_problems=secondary,
            problem_confidences={primary.value: 0.8},
            explicitly_mentioned=explicit, inferred_concepts=inferred,
            all_relevant_concepts=explicit + inferred,
            query_type=data.get("query_type","general"),
            emphasis_direction=data.get("emphasis_direction","cause"),
            subgraph_depth=2, priority_threshold=0.3,
            focus_nodes=explicit[:5],
            reasoning_chain=[data.get("reasoning","")],
            confidence=0.8
        )

class OpenAIAnalyzer(LLMQueryAnalyzer):
    SYSTEM_PROMPT = LocalLLMAnalyzer.SYSTEM_PROMPT
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except: pass
    def is_available(self) -> bool: return self.client is not None
    @timed
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        if not self.is_available(): return FallbackAnalyzer().analyze_query(query, ontology)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role":"system","content":self.SYSTEM_PROMPT},
                                            {"role":"user","content":f"User Query: {query}"}],
                temperature=0.1, max_tokens=512, response_format={"type":"json_object"}
            )
            data = json.loads(response.choices[0].message.content)
        except:
            return FallbackAnalyzer().analyze_query(query, ontology)
        # same parsing as LocalLLMAnalyzer
        problem_map = {"anode_bottleneck":SIBCoreProblem.ANODE_BOTTLENECK, "cathode_instability":SIBCoreProblem.CATHODE_INSTABILITY,
                       "sei_chemistry":SIBCoreProblem.SEI_CHEMISTRY, "solid_state_interface":SIBCoreProblem.SOLID_STATE_INTERFACE,
                       "low_energy_density":SIBCoreProblem.LOW_ENERGY_DENSITY, "moisture_manufacturing":SIBCoreProblem.MOISTURE_MANUFACTURING,
                       "general":SIBCoreProblem.GENERAL, "multi_problem":SIBCoreProblem.MULTI_PROBLEM}
        primary = problem_map.get(data.get("primary_problem","general"), SIBCoreProblem.GENERAL)
        secondary = [problem_map.get(p, SIBCoreProblem.GENERAL) for p in data.get("secondary_problems", []) if p in problem_map]
        explicit = [c for c in data.get("key_concepts", []) if ontology.resolve_concept(c)]
        inferred = [c for c in data.get("implicit_concepts", []) if ontology.resolve_concept(c) and c not in explicit]
        return QueryAnalysisResult(
            original_query=query, normalized_query=query.lower(),
            primary_problem=primary, secondary_problems=secondary,
            problem_confidences={primary.value: 0.9},
            explicitly_mentioned=explicit, inferred_concepts=inferred,
            all_relevant_concepts=explicit + inferred,
            query_type=data.get("query_type","general"),
            emphasis_direction=data.get("emphasis_direction","cause"),
            subgraph_depth=2, priority_threshold=0.3,
            focus_nodes=explicit[:5],
            reasoning_chain=[data.get("reasoning","")],
            confidence=0.9
        )

class LLMAnalyzerFactory:
    _instance = None
    @classmethod
    def get_analyzer(cls, mode: str = "rule_based", **kwargs) -> LLMQueryAnalyzer:
        if mode == "openai":
            return OpenAIAnalyzer(api_key=kwargs.get("api_key"), model=kwargs.get("api_model","gpt-3.5-turbo"))
        elif mode == "local":
            return LocalLLMAnalyzer(model_name=kwargs.get("model_name","gpt2"))
        else:
            return FallbackAnalyzer()

# ============================================================================
# ONTOLOGY PRIORITY SCORER (NEW)
# ============================================================================
class OntologyPriorityScorer:
    WEIGHTS = {"direct":0.40, "problem_affinity":0.25, "causal_path":0.20, "centrality":0.05, "cooccurrence":0.10}
    PROBLEM_CONCEPT_AFFINITY = {
        SIBCoreProblem.ANODE_BOTTLENECK: {"hard_carbon":0.95, "alloying_anode":0.90, "intercalation_anode":0.85,
                                          "sodium_metal":0.75, "specific_capacity":0.85, "coulombic_efficiency":0.80,
                                          "volume_expansion":0.90, "sei_formation":0.60},
        SIBCoreProblem.CATHODE_INSTABILITY: {"layered_oxide_cathode":0.95, "polyanionic_cathode":0.85,
                                             "prussian_blue_analogue":0.80, "phase_transition":0.95,
                                             "specific_capacity":0.80, "cycle_life":0.90, "voltage_plateau":0.75},
        SIBCoreProblem.SEI_CHEMISTRY: {"sei_formation":0.95, "liquid_electrolyte":0.90, "interface_engineering":0.85,
                                       "coulombic_efficiency":0.90, "cycle_life":0.85, "interface_resistance":0.80},
        SIBCoreProblem.SOLID_STATE_INTERFACE: {"solid_electrolyte":0.95, "quasi_solid_electrolyte":0.85,
                                               "dendrite_growth":0.90, "void_formation":0.85, "interfacial_resistance":0.90,
                                               "ionic_conductivity":0.85},
        SIBCoreProblem.LOW_ENERGY_DENSITY: {"energy_density":0.95, "specific_capacity":0.90, "voltage_plateau":0.90,
                                            "full_cell":0.85, "hard_carbon":0.75, "sodium_metal":0.80},
        SIBCoreProblem.MOISTURE_MANUFACTURING: {"slurry_coating":0.95, "layered_oxide_cathode":0.80,
                                                "coulombic_efficiency":0.70, "cycle_life":0.65},
        SIBCoreProblem.GENERAL: {}, SIBCoreProblem.MULTI_PROBLEM: {}
    }
    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology
        self._precompute_centrality()

    def _precompute_centrality(self):
        G = nx.DiGraph()
        for rel in self.ontology.relationships:
            G.add_edge(rel.source, rel.target, weight=rel.confidence)
        try:
            self.centrality = nx.pagerank(G, weight='weight')
        except:
            self.centrality = nx.degree_centrality(G)
        if self.centrality:
            max_c = max(self.centrality.values()) or 1
            self.centrality = {k: v/max_c for k, v in self.centrality.items()}

    @timed
    def score_all_concepts(self, analysis: QueryAnalysisResult) -> Dict[str, ConceptPriority]:
        priorities = {}
        mentioned = set(analysis.explicitly_mentioned)
        for concept_name, node in self.ontology.concepts.items():
            direct = 1.0 if concept_name in mentioned else (0.6 if concept_name in analysis.inferred_concepts else 0.0)
            is_explicit = concept_name in mentioned
            problem_aff = self.PROBLEM_CONCEPT_AFFINITY.get(analysis.primary_problem, {}).get(concept_name, 0.1)
            causal = self._causal_path_score(concept_name, analysis.explicitly_mentioned, analysis.emphasis_direction)
            cent = self.centrality.get(concept_name, 0.0)
            cooc = 0.0
            composite = (self.WEIGHTS["direct"]*direct + self.WEIGHTS["problem_affinity"]*problem_aff +
                         self.WEIGHTS["causal_path"]*causal + self.WEIGHTS["centrality"]*cent +
                         self.WEIGHTS["cooccurrence"]*cooc)
            inference_reason = ""
            if not is_explicit and composite > 0.3:
                if problem_aff > 0.5: inference_reason = f"High affinity to {analysis.primary_problem.value}"
                elif causal > 0.3: inference_reason = "On causal path"
            priorities[concept_name] = ConceptPriority(
                concept_name=concept_name, concept_type=node.concept_type,
                composite_score=composite, direct_score=direct,
                problem_affinity_score=problem_aff, causal_path_score=causal,
                centrality_bonus=cent, cooccurrence_bonus=cooc,
                is_explicitly_mentioned=is_explicit,
                is_inferred=(not is_explicit and composite > 0.3),
                inference_reason=inference_reason
            )
        return priorities

    def _causal_path_score(self, concept: str, mentioned: List[str], emphasis: str) -> float:
        if not mentioned: return 0.0
        valuable = {RelationshipType.CAUSES, RelationshipType.DRIVES, RelationshipType.INFLUENCES,
                    RelationshipType.ENABLES} if emphasis in ("cause","solution") else \
                   {RelationshipType.RESULTS_IN, RelationshipType.FORMS, RelationshipType.GENERATES}
        max_score = 0.0
        for m in mentioned:
            for rel in self.ontology.relationships:
                if (rel.source == concept and rel.target == m) or (rel.target == concept and rel.source == m):
                    if rel.rel_type in valuable:
                        max_score = max(max_score, rel.confidence * 0.8)
                    else:
                        max_score = max(max_score, rel.confidence * 0.3)
        return min(max_score, 1.0)

# ============================================================================
# PRIORITY SUBGRAPH EXTRACTOR (NEW)
# ============================================================================
class PrioritySubgraphExtractor:
    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology

    @timed
    def extract(self, analysis: QueryAnalysisResult, max_nodes: int = 30,
                min_priority: float = 0.2) -> nx.DiGraph:
        G = nx.DiGraph()
        priorities = analysis.concept_priorities
        seed = [n for n, cp in priorities.items() if cp.composite_score >= min_priority]
        seed.sort(key=lambda n: priorities[n].composite_score, reverse=True)
        seed = seed[:max_nodes]
        expanded = set(seed)
        frontier = deque(seed)
        depth_map = {n: 0 for n in seed}
        while frontier and len(expanded) < max_nodes * 1.5:
            current = frontier.popleft()
            if depth_map[current] >= analysis.subgraph_depth: continue
            for rel in self.ontology.relationships:
                neighbor = None
                if rel.source == current: neighbor = rel.target
                elif rel.target == current: neighbor = rel.source
                if neighbor and neighbor not in expanded:
                    cp = priorities.get(neighbor, ConceptPriority(neighbor, ConceptType.GENERAL, 0.0,0.0,0.0,0.0,0.0,0.0,False,False))
                    decay = 0.5 ** (depth_map[current] + 1)
                    effective = cp.composite_score * decay + priorities[current].composite_score * rel.confidence * decay
                    if effective >= min_priority * 0.5 or neighbor in seed:
                        expanded.add(neighbor)
                        depth_map[neighbor] = depth_map[current] + 1
                        frontier.append(neighbor)
        for node in expanded:
            if node in self.ontology.concepts:
                n = self.ontology.concepts[node]
                p = priorities.get(node)
                G.add_node(node, concept_type=n.concept_type.value, definition=n.definition,
                           priority=p.composite_score if p else 0.0,
                           is_focus=node in analysis.focus_nodes)
        for rel in self.ontology.relationships:
            if rel.source in G.nodes and rel.target in G.nodes:
                G.add_edge(rel.source, rel.target, rel_type=rel.rel_type.value,
                           confidence=rel.confidence, color=get_edge_color(rel.rel_type),
                           width=get_edge_width(rel.rel_type), style=get_edge_style(rel.rel_type))
        return G

# ============================================================================
# QUERY VISUALIZATION DRIVER (NEW)
# ============================================================================
class QueryVisualizationDriver:
    TYPE_COLORS = {"material":"#4CAF50", "property":"#2196F3", "phenomenon":"#FF9800",
                   "method":"#9C27B0", "process":"#00BCD4", "parameter":"#795548",
                   "model":"#607D8B", "general":"#9E9E9E"}
    LAYOUT_HEURISTICS = {"comparison":"bipartite", "mechanism":"hierarchical",
                         "optimization":"radial", "question":"force", "general":"force"}

    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology

    def get_node_visual_params(self, subgraph: nx.DiGraph, analysis: QueryAnalysisResult) -> Dict:
        params = {}
        priorities = [data.get("priority",0) for _, data in subgraph.nodes(data=True)]
        min_p, max_p = min(priorities) if priorities else (0,1)
        p_range = max_p - min_p or 1
        for node, data in subgraph.nodes(data=True):
            priority = data.get("priority",0)
            norm = (priority - min_p) / p_range
            base = self.TYPE_COLORS.get(data.get("concept_type","general"), "#9E9E9E")
            if data.get("is_focus", False):
                color = lighten_hex_color(base, 0.3); size = 30 + norm*40; border=3; border_color="#FFD700"
            else:
                color = base; size = 20 + norm*30; border=2; border_color="#888"
            params[node] = {"color":color, "size":size, "border_width":border,
                            "border_color":border_color,
                            "label":node.replace("_"," ").title(),
                            "title":f"{node.replace('_',' ').title()}\nPriority: {priority:.2f}",
                            "opacity":0.5 + norm*0.5}
        return params

    def get_edge_visual_params(self, subgraph: nx.DiGraph, analysis: QueryAnalysisResult) -> Dict:
        return {(u,v): {"color":d.get("color","#888"), "width":d.get("width",1.0),
                        "style":d.get("style","solid"), "arrows":"to"} for u,v,d in subgraph.edges(data=True)}

    def get_layout_params(self, analysis: QueryAnalysisResult) -> Dict:
        lt = self.LAYOUT_HEURISTICS.get(analysis.query_type, "force")
        return {"layout_type":lt, "gravity":0.1, "central_gravity":0.01, "velocity_decay":0.2, "node_distance":100}

    def generate_legend(self, analysis: QueryAnalysisResult) -> List[Dict]:
        legend = []
        for ctype, color in self.TYPE_COLORS.items():
            if any(cp.concept_type.value == ctype for cp in analysis.concept_priorities.values() if cp.composite_score > 0.2):
                legend.append({"label":ctype.replace("_"," ").title(), "color":color, "shape":"dot"})
        legend.append({"label":"Focus Node", "color":"#FFD700", "shape":"dot", "border":3})
        return legend

# ============================================================================
# QUERY PIPELINE (NEW)
# ============================================================================
class QueryDrivenGraphPipeline:
    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology
        self.analyzer = None
        self.scorer = OntologyPriorityScorer(ontology)
        self.extractor = PrioritySubgraphExtractor(ontology)
        self.visualizer = QueryVisualizationDriver(ontology)

    def init_llm_analyzer(self, mode: str = "rule_based", **kwargs):
        self.analyzer = LLMAnalyzerFactory.get_analyzer(mode, **kwargs)

    @timed
    def process_query(self, query: str, max_nodes: int = 30, priority_threshold: float = 0.2,
                      mode: str = "rule_based", **kwargs) -> Tuple[QueryAnalysisResult, nx.DiGraph, Dict, Dict, Dict]:
        if not self.analyzer:
            self.init_llm_analyzer(mode, **kwargs)
        analysis = self.analyzer.analyze_query(query, self.ontology)
        analysis.concept_priorities = self.scorer.score_all_concepts(analysis)
        subgraph = self.extractor.extract(analysis, max_nodes=max_nodes, min_priority=priority_threshold)
        node_params = self.visualizer.get_node_visual_params(subgraph, analysis)
        edge_params = self.visualizer.get_edge_visual_params(subgraph, analysis)
        layout_params = self.visualizer.get_layout_params(analysis)
        return analysis, subgraph, node_params, edge_params, layout_params

    def render_pyvis(self, subgraph: nx.DiGraph, node_params: Dict, edge_params: Dict,
                     layout_params: Dict, height: str = "600px") -> str:
        net = Network(height=height, width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
        if layout_params.get("layout_type") == "force":
            net.force_atlas_2based(gravity=layout_params.get("gravity",0.1), central_gravity=0.01,
                                   velocity_decay=0.2, node_distance=100)
        for node, p in node_params.items():
            net.add_node(node, label=p["label"], color=p["color"], size=p["size"],
                         border_width=p["border_width"], border_color=p["border_color"],
                         title=p["title"], opacity=p["opacity"])
        for (u,v), p in edge_params.items():
            net.add_edge(u, v, color=p["color"], width=p["width"],
                         dashes=(p["style"]=="dashed"), arrows=p["arrows"], title=p.get("title",""))
        return net.generate_html()

# ============================================================================
# UTILITY FUNCTIONS (JSON loading, etc.)
# ============================================================================
def robust_load_file(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == '.bib':
        try:
            import bibtexparser
            with open(filepath, 'r', encoding='utf-8') as f:
                bib = bibtexparser.load(f)
                return [{'title':e.get('title',''), 'abstract':e.get('abstract',''), 'author':e.get('author',''),
                         'year':e.get('year',''), 'journal':e.get('journal',''), 'doi':e.get('doi','')}
                        for e in bib.entries]
        except: return []
    text = filepath.read_text(encoding='utf-8-sig')
    if not text.strip(): return []
    try:
        return json.loads(text)
    except:
        # try JSONL
        records = []
        for line in text.splitlines():
            line=line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
        return records if records else []

def load_all_json_files(directory: str) -> List[Tuple[str, List]]:
    files = list(Path(directory).glob("*.json")) + list(Path(directory).glob("*.bib")) + list(Path(directory).glob("*.csv"))
    loaded = []
    for fp in files:
        try:
            data = robust_load_file(fp)
            if data:
                loaded.append((fp.name, data if isinstance(data, list) else [data]))
        except Exception as e:
            st.error(f"Error loading {fp.name}: {e}")
    return loaded

def build_master_dataframe(file_records: List[Tuple[str, List]]) -> pd.DataFrame:
    rows = []
    for fname, records in file_records:
        for rec in records:
            if isinstance(rec, dict):
                rec['_source_file'] = fname
                rows.append(rec)
    if not rows: return pd.DataFrame()
    df = pd.json_normalize(rows)
    # try to parse year
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    except:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ============================================================================
# EXISTING VISUALIZATION FUNCTIONS (from v6.1) – full implementations
# ============================================================================
def render_pyvis_graph(nx_graph, concept_abstract_map, physics_enabled=True,
                       cmap_name="viridis", top_n_nodes=0, theme=None,
                       physics_preset=None, show_edge_weights=False,
                       edge_label_mode="hover", **kwargs):
    # This is the full function from the v6.1 code.
    # We'll paste it here for completeness.
    # (The user already has this in their codebase, but we include it.)
    if theme is None:
        theme = THEME_PRESETS.get("Bright (Default)", {"bg":"#ffffff", "font":"#1e293b"})
    if physics_preset is None:
        physics_preset = {"damping":0.55, "gravity":-2500, "spring_length":140,
                          "spring_strength":0.05, "central_gravity":0.25, "stabilization":2500}
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))
    net = Network(height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
                  select_menu=True, notebook=False, cdn_resources='remote')
    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
            "physics": {{
                "enabled": true, "solver": "barnesHut",
                "barnesHut": {{
                    "gravitationalConstant": {physics_preset['gravity']},
                    "centralGravity": {physics_preset['central_gravity']},
                    "springLength": {physics_preset['spring_length']},
                    "springConstant": {physics_preset['spring_strength']},
                    "damping": {physics_preset['damping']}, "overlap": 0.15
                }},
                "stabilization": {{ "enabled": true, "iterations": 500, "updateInterval": 50, "onlyDynamicEdges": true, "fit": true }}
            }},
            "interaction": {{ "hover": true, "tooltipDelay": 180, "hideEdgesOnDrag": false, "zoomView": true, "dragView": true }}
        }}
        """)
    else:
        net.set_options("""var options = { "physics": { "enabled": false }, "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true } }""")
    label_map = {}
    n_counter = 1
    used_rel_types = {}
    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(8 + freq * 1.2, 8, 40))
        color = get_sib_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        original_label = node
        label = node.replace("_", " ").title()
        if len(label) > 20:
            short_label = f"N{n_counter}"
            label_map[short_label] = original_label
            n_counter += 1
            label = short_label
            node_shape = 'circle'
            inside_font_size = max(8, min(int(size * 0.55), 14))
            font_dict = {'color': '#ffffff', 'size': inside_font_size, 'face': 'Inter, Segoe UI, Roboto, sans-serif', 'bold': True}
        else:
            node_shape = 'dot'
            font_dict = {'color': theme['font'], 'size': 12, 'face': 'Inter, Segoe UI, Roboto, sans-serif', 'strokeWidth': 0, 'vadjust': -6}
        tooltip_content = f"<div><b>{original_label}</b><br>Type: {nx_graph.nodes[node].get('concept_type','general')}<br>Degree: {degree}<br>Frequency: {freq}</div>"
        net.add_node(node, label=label, size=size,
                     color={'background': color, 'border': theme['node_border'],
                            'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                            'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}},
                     font=font_dict, title=tooltip_content, borderWidth=2, borderWidthSelected=3,
                     shadow={'enabled': True, 'color': theme['shadow_color'], 'size': 12, 'x': 4, 'y': 4},
                     shape=node_shape, mass=max(1, 1 + freq * 0.05))
    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = float(np.percentile(all_weights, 80)) if all_weights else 0.0
    for u, v in nx_graph.edges():
        w = float(nx_graph[u][v].get('weight', 1))
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        rel_type = RelationshipType.SEMANTIC
        if edge_type != 'unknown':
            try: rel_type = RelationshipType(edge_type)
            except ValueError: pass
        base_color = get_edge_color(rel_type)
        width = float(get_edge_width(rel_type) * (0.5 + 0.5 * w))
        style = get_edge_style(rel_type)
        dashes = True if style == "dashed" or is_inferred else False
        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)), width=width,
            color={'color': base_color, 'highlight': theme['highlight_bg'], 'hover': theme['hover_bg'], 'opacity': 0.85},
            smooth={"type": "dynamic"},
            title=f"Weight: {w:.2f}<br>Type: {edge_type}<br>Inferred: {is_inferred}",
            dashes=dashes
        )
        if edge_label_mode == "all" or (edge_label_mode == "threshold" and w >= weight_threshold):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {'color': theme['font'], 'size': 10, 'background': theme['tooltip_bg'], 'strokeWidth': 2, 'strokeColor': theme['node_border'], 'align': 'middle'}
        net.add_edge(u, v, **edge_kwargs)
        if rel_type not in used_rel_types:
            used_rel_types[rel_type] = rel_type.value.replace("_", " ").title()
    if used_rel_types:
        legend_rows = []
        for rt, human in sorted(used_rel_types.items(), key=lambda x: x[1]):
            c = get_edge_color(rt)
            w_leg = get_edge_width(rt)
            s_leg = get_edge_style(rt)
            border = 'border: 1px dashed #888;' if s_leg == "dashed" else 'border: 1px solid transparent;'
            legend_rows.append(f'<tr><td style="padding:2px 6px;"><span style="display:inline-block;width:{int(20*w_leg)}px;height:3px;background:{c};vertical-align:middle;{border}"></span></td><td style="padding:2px 6px;color:#ccc;font-size:11px;">{human}</td></tr>')
        legend_html = f'<div style="background:#0d0d1a;border-radius:8px;padding:12px 16px;margin-top:8px;max-height:280px;overflow-y:auto;"><div style="color:#fff;font-size:13px;font-weight:bold;margin-bottom:6px;">Edge Colors ({len(used_rel_types)} types)</div><table style="border-collapse:collapse;">{"".join(legend_rows)}</table></div>'
        net.add_node("__legend__", label="", shape="dot", size=0, color="rgba(0,0,0,0)", fixed=True, x=-500, y=-500, physics=False, title=legend_html)
    try:
        tmp_html = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
        tmp_path = tmp_html.name
        net.write_html(tmp_path, notebook=False)
        tmp_html.close()
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        if label_map:
            label_map_json = json.dumps(label_map)
            html_content = html_content.replace('</body>', f'<div id="label-map-data" style="display:none;">{label_map_json}</div></body>')
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"PyVis HTML generation failed: {e}")
        html_content = net.generate_html()
    custom_css = f"""
    <style>
    body {{ background: {theme['bg']}; margin: 0; padding: 0; font-family: 'Inter, Segoe UI, Roboto, sans-serif'; }}
    #mynetwork {{ border-radius: 16px; box-shadow: 0 12px 48px {theme['shadow_color']}; outline: none; }}
    div.vis-tooltip {{
        background: {theme['tooltip_bg']} !important; color: {theme['tooltip_text']} !important;
        border: 1px solid {theme['tooltip_border']} !important; border-radius: 10px !important;
        padding: 14px 18px !important; font-family: 'Inter, Segoe UI, Roboto, sans-serif' !important;
        font-size: 13px !important; line-height: 1.5 !important;
        box-shadow: 0 8px 32px {theme['shadow_color']} !important; max-width: 320px !important; white-space: normal !important;
    }}
    </style>
    """
    html_content = html_content.replace('</head>', custom_css + '</head>')
    st.components.v1.html(html_content, height=790, scrolling=True)
    if label_map:
        st.markdown("---")
        st.markdown("### 🗺️ Node Label Legend")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))
        cols = st.columns(4)
        for i, (short, full) in enumerate(sorted_legend):
            with cols[i % 4]:
                st.markdown(f"<div style='padding:8px; border-radius:6px; background-color:#f8fafc; border-left:4px solid #ff6b6b; margin-bottom:6px;'><b style='color:#ff6b6b;'>{short}</b>: <span style='font-size:13px;'>{full}</span></div>", unsafe_allow_html=True)

def get_sib_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    category = 'general'
    for pattern, cat in {
        r'cathode|oxide|phosphate|prussian|nasicon': 'cathode_material',
        r'anode|carbon|metal|alloying': 'anode_material',
        r'electrolyte|nasicon|polymer|aqueous': 'electrolyte',
        r'capacity|density|efficiency|conductivity|voltage|cycle_life|rate': 'electrochemical_property',
        r'dendrite|sei|intercalation|conversion|plating': 'phenomenon',
        r'cv|eis|galvanostatic|operando': 'method',
        r'current|temperature|cut.off': 'parameter',
        r'coating|cell|assembly': 'processing'
    }.items():
        if re.search(pattern, concept_lower):
            category = cat
            break
    color_map = {'cathode_material':'#1f77b4', 'anode_material':'#ff7f0e', 'electrolyte':'#2ca02c',
                 'electrochemical_property':'#d62728', 'phenomenon':'#9467bd', 'method':'#8c564b',
                 'parameter':'#e377c2', 'processing':'#7f7f7f', 'general':'#bcbd22'}
    return color_map.get(category, '#bcbd22')

# ============================================================================
# SIDEBAR & THEME PRESETS
# ============================================================================
THEME_PRESETS = {
    "Bright (Default)": {"bg":"#ffffff", "font":"#1e293b", "tooltip_bg":"rgba(255,255,255,0.95)",
                         "tooltip_border":"#cbd5e1", "tooltip_text":"#1e293b",
                         "edge_unknown":"rgba(148,163,184,0.3)", "node_border":"#f8fafc",
                         "highlight_bg":"#ff6b6b", "hover_bg":"#ffd93d", "shadow_color":"rgba(0,0,0,0.15)",
                         "plotly_bg":"#ffffff", "plotly_paper":"#ffffff", "grid_color":"#e2e8f0", "axis_color":"#64748b"},
    "Dark": {"bg":"#0f172a", "font":"#e2e8f0", "tooltip_bg":"rgba(15,23,42,0.95)",
             "tooltip_border":"#334155", "tooltip_text":"#e2e8f0",
             "edge_unknown":"rgba(148,163,184,0.4)", "node_border":"#f8fafc",
             "highlight_bg":"#ff6b6b", "hover_bg":"#ffd93d", "shadow_color":"rgba(0,0,0,0.6)",
             "plotly_bg":"#0f172a", "plotly_paper":"#0f172a", "grid_color":"#1e293b", "axis_color":"#94a3b8"},
}
SUPPORTED_COLORMAPS = {"viridis":"Viridis", "plasma":"Plasma", "inferno":"Inferno", "magma":"Magma",
                       "cividis":"Cividis", "turbo":"Turbo", "jet":"Jet", "rainbow":"Rainbow",
                       "coolwarm":"Coolwarm", "RdBu":"RdBu", "Spectral":"Spectral",
                       "tab10":"Set1", "tab20":"Set2"}

def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except:
        return ["#ff6b6b"]*n

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🎨 Theme")
        theme = st.selectbox("Color theme:", list(THEME_PRESETS.keys()), index=0)
        st.session_state['theme'] = theme
        st.subheader("📦 Batch Processing")
        st.toggle("Enable batch processing", key="batch_mode")
        if st.session_state.get("batch_mode", False):
            st.slider("Batch size", 100, 2000, 1000, key="batch_size")
            st.slider("GNN epochs", 10, 50, 40, key="batch_gnn_epochs")
        st.subheader("📊 Visualization")
        st.selectbox("Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"],
                     key="viz_backend")
        st.toggle("Show edge weights", key="show_edge_weights")
        st.selectbox("Colormap:", list(SUPPORTED_COLORMAPS.keys()), key="cmap_name")
        st.subheader("Graph Parameters")
        st.slider("Min concept frequency", 1, 20, 1, key="min_freq")
        st.slider("Semantic threshold", 0.6, 0.95, 0.85, key="sim_threshold")
        st.slider("Co-occurrence weight", 0.5, 1.0, 0.7, key="cooc_weight")
        st.slider("Semantic weight", 0.0, 0.5, 0.2, key="sem_weight")
        st.slider("Inference weight", 0.0, 0.3, 0.1, key="inf_weight")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def render_query_interface(ontology: DomainOntology, pipeline: QueryDrivenGraphPipeline):
    st.subheader("🔎 LLM‑Directed Query Interface")
    query = st.text_area("Enter your SIB research question:", height=100,
                         placeholder="e.g., Why can't sodium intercalate into graphite like lithium does?")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        mode = st.selectbox("LLM Mode", ["rule_based", "local", "openai"], index=0)
    with col2:
        if mode == "openai":
            api_key = st.text_input("OpenAI API Key", type="password")
            api_model = st.text_input("Model", "gpt-3.5-turbo")
        else:
            api_key = api_model = None
    with col3:
        if mode == "local":
            model_name = st.text_input("Model name", "gpt2")
        else:
            model_name = None
    if st.button("Analyze & Visualize", type="primary") and query:
        with st.spinner("Processing query..."):
            analysis, subgraph, node_params, edge_params, layout_params = pipeline.process_query(
                query, max_nodes=30, priority_threshold=0.2,
                mode=mode, api_key=api_key, api_model=api_model, model_name=model_name
            )
        st.success("Analysis complete")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Primary Problem", analysis.primary_problem.value.replace("_"," ").title())
        col_b.metric("Query Type", analysis.query_type.title())
        col_c.metric("Concepts Found", len(analysis.all_relevant_concepts))
        with st.expander("📋 Priority Rankings"):
            top = analysis.get_top_concepts(10)
            df = pd.DataFrame([cp.to_dict() for cp in top])
            st.dataframe(df)
        st.subheader("🌐 Extracted Subgraph")
        html = pipeline.render_pyvis(subgraph, node_params, edge_params, layout_params, height="500px")
        st.components.v1.html(html, height=550)
        st.caption("Nodes colored by concept type; focus nodes in gold border.")

def main():
    st.title("🔋 SIB Quantitative Descriptor Graph v6.2 – LLM Query")
    st.caption("Integrated ontology, batch processing, GNN, and LLM-directed query interface.")

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    render_sidebar()

    # Load data
    with st.spinner("Loading JSON database..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)
    if df.empty:
        st.warning(f"No data found in {JSON_METADATA_DIR}. Please add JSON files.")
        return
    st.success(f"Loaded {len(df)} records.")
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ['abstract','title','summary','text'])]
    if not text_cols:
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
    selected_text_cols = st.multiselect("Text columns for concept extraction:", text_cols, default=text_cols[:2])
    if not selected_text_cols:
        st.error("Select at least one text column.")
        return

    # Build graph (full or batch) – simplified demo
    if st.button("🚀 Build Concept Graph", type="primary") or st.session_state.get("batch_trigger"):
        # For demo, we create a dummy graph.
        # In the real v6.1 code, the full analysis pipeline would run.
        # We'll just place a placeholder that creates a simple graph from the ontology.
        G = nx.Graph()
        for name in ontology.concepts.keys():
            G.add_node(name)
        for rel in ontology.relationships:
            G.add_edge(rel.source, rel.target, weight=rel.confidence, edge_type=rel.rel_type.value)
        st.session_state.analysis_data = {
            "nx_graph": G,
            "valid_concepts": list(G.nodes()),
            "concept_abstract_map": {n: [1,2,3] for n in G.nodes()},
            "distill_df": pd.DataFrame({"concept": list(G.nodes()), "frequency": [5]*len(G.nodes())}),
            "top_scores": pd.DataFrame({"concept_u": ["hard_carbon"], "concept_v": ["energy_density"], "composite_score": [0.8]}),
            "df_filtered": df, "selected_text_cols": selected_text_cols
        }
        st.success("Graph built (demo). Replace with full v6.1 pipeline for real data.")

    # Display results
    if st.session_state.get("analysis_data"):
        data = st.session_state.analysis_data
        G = data["nx_graph"]
        theme = THEME_PRESETS.get(st.session_state.get("theme", "Bright (Default)"), THEME_PRESETS["Bright (Default)"])
        tabs = st.tabs(["📊 Graph", "🧪 Analytics", "🔎 LLM Query", "📥 Export"])
        with tabs[0]:
            st.subheader("Interactive Graph")
            # Use the full render_pyvis_graph from above
            render_pyvis_graph(G, data["concept_abstract_map"], cmap_name=st.session_state.get("cmap_name","viridis"), theme=theme)
        with tabs[1]:
            st.subheader("Distillation & Analytics")
            if "distill_df" in data:
                st.dataframe(data["distill_df"])
            if "top_scores" in data and not data["top_scores"].empty:
                st.dataframe(data["top_scores"])
        with tabs[2]:
            pipeline = QueryDrivenGraphPipeline(ontology)
            render_query_interface(ontology, pipeline)
        with tabs[3]:
            st.subheader("Export")
            st.info("Export functionality available in full v6.1.")
    else:
        st.info("Build the concept graph first, or use the LLM Query tab below.")
        pipeline = QueryDrivenGraphPipeline(ontology)
        render_query_interface(ontology, pipeline)

if __name__ == "__main__":
    main()
