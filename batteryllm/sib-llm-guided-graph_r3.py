#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIB QUANTITATIVE DESCRIPTOR GRAPH v6.2 – LLM‑GUIDED DYNAMIC ONTOLOGY
====================================================================
Complete integration of:
- v6.1: Memory‑safe batch processing, ontology, GNN, analytics, visualizations
- NEW: LLM‑directed query analysis (OpenAI/local/fallback)
- NEW: Dynamic ontology expansion (add concepts, relationships, bridges)
- NEW: Priority scoring and subgraph extraction
- NEW: Query‑driven visualisation with highlighting

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
# QUERY ANALYSIS DATA STRUCTURES (NEW)
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

        # Explicitly mentioned concepts
        explicitly_mentioned = []
        for canonical, node in ontology.concepts.items():
            if canonical.replace("_", " ") in q:
                explicitly_mentioned.append(canonical)
                continue
            for syn in node.synonyms:
                if syn.replace("_", " ") in q:
                    explicitly_mentioned.append(canonical)
                    break

        # Infer additional concepts from problem definitions
        inferred = []
        if primary != SIBCoreProblem.GENERAL:
            problem_def = SIB_PROBLEM_DEFINITIONS[primary]
            for concept in problem_def.key_concepts:
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)
            for concept in problem_def.relevant_materials:
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)
            for concept in problem_def.relevant_phenomena:
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)
            for concept in problem_def.relevant_properties:
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)

        all_relevant = list(dict.fromkeys(explicitly_mentioned + inferred))

        # Build priority scores
        priorities = {}
        problem_def = SIB_PROBLEM_DEFINITIONS.get(primary, SIB_PROBLEM_DEFINITIONS[SIBCoreProblem.GENERAL])
        problem_concept_set = problem_def.get_ontology_concepts()

        for concept in all_relevant:
            is_explicit = concept in explicitly_mentioned
            direct_score = 1.0 if is_explicit else 0.6
            problem_affinity = 1.0 if concept in problem_concept_set else 0.4
            causal_score = 0.5
            for rel in ontology.relationships:
                if rel.source == concept or rel.target == concept:
                    if rel.source in problem_concept_set and rel.target in problem_concept_set:
                        causal_score = max(causal_score, rel.confidence)
            composite = (direct_score * 0.35 + problem_affinity * 0.35 +
                         causal_score * 0.20 + 0.10)
            priorities[concept] = ConceptPriority(
                concept_name=concept,
                concept_type=ontology.get_concept_type(concept),
                composite_score=composite,
                direct_score=direct_score,
                problem_affinity_score=problem_affinity,
                causal_path_score=causal_score,
                centrality_bonus=0.0,
                cooccurrence_bonus=0.0,
                is_explicitly_mentioned=is_explicit,
                is_inferred=not is_explicit,
                inference_reason="problem_affinity" if not is_explicit else "explicit_mention"
            )

        # Detect query type
        query_type = "general"
        emphasis = "cause"
        comparison_pairs = []
        if any(w in q for w in ["compare", "vs", "versus", "difference", "contrast", "between"]):
            query_type = "comparison"
            emphasis = "neutral"
            mentioned = [c.replace("_", " ") for c in explicitly_mentioned]
            for i in range(len(mentioned)):
                for j in range(i + 1, len(mentioned)):
                    comparison_pairs.append((mentioned[i], mentioned[j]))
        elif any(w in q for w in ["why", "cause", "reason", "lead to", "result in", "due to"]):
            query_type = "causal"
            emphasis = "cause"
        elif any(w in q for w in ["how", "improve", "enhance", "optimize", "strategy", "solution", "approach"]):
            query_type = "solution"
            emphasis = "effect"
        elif any(w in q for w in ["what is", "define", "describe", "explain", "meaning"]):
            query_type = "definition"
            emphasis = "neutral"

        # Build highlight paths from problem key_relationships
        highlight_paths = []
        if primary != SIBCoreProblem.GENERAL:
            for src, rel_str, tgt in problem_def.key_relationships:
                src_resolved = ontology.resolve_concept(src)
                tgt_resolved = ontology.resolve_concept(tgt)
                if src_resolved and tgt_resolved:
                    highlight_paths.append([src_resolved, tgt_resolved])

        # Reasoning chain
        reasoning = [f"Query normalized: '{q}'"]
        reasoning.append(f"Primary problem identified: {primary.value} (score: {problem_scores.get(primary, 0)})")
        if secondary:
            reasoning.append(f"Secondary problems: {[p.value for p in secondary]}")
        reasoning.append(f"Explicitly mentioned concepts: {len(explicitly_mentioned)}")
        reasoning.append(f"Inferred concepts from problem context: {len(inferred)}")
        reasoning.append(f"Query type: {query_type}, emphasis: {emphasis}")

        # Normalize problem scores
        total = max(sum(problem_scores.values()), 1)
        problem_confidences = {p.value: s / total for p, s in problem_scores.items()}

        return QueryAnalysisResult(
            original_query=query,
            normalized_query=q,
            primary_problem=primary,
            secondary_problems=secondary,
            problem_confidences=problem_confidences,
            explicitly_mentioned=explicitly_mentioned,
            inferred_concepts=inferred,
            all_relevant_concepts=all_relevant,
            concept_priorities=priorities,
            query_type=query_type,
            emphasis_direction=emphasis,
            comparison_pairs=comparison_pairs,
            subgraph_depth=2,
            priority_threshold=0.3,
            focus_nodes=explicitly_mentioned[:5],
            bridge_nodes=inferred[:3],
            suggested_layout="force" if query_type != "comparison" else "bisected",
            highlight_paths=highlight_paths,
            visualization_focus=problem_def.visualization_focus,
            reasoning_chain=reasoning,
            confidence=min(sum(problem_scores.values()) / 3.0, 1.0),
        )

class OpenAIQueryAnalyzer(LLMQueryAnalyzer):
    """Uses OpenAI API for sophisticated query understanding."""

    SYSTEM_PROMPT = """You are an expert Sodium-Ion Battery (SIB) researcher. Analyze the user's query and return a JSON object with:

1. "primary_problem": One of: anode_bottleneck, cathode_instability, sei_chemistry, solid_state_interface, low_energy_density, moisture_manufacturing, general, multi_problem
2. "secondary_problems": List of problem IDs (can be empty)
3. "explicitly_mentioned": List of canonical concept names from the query (use snake_case)
4. "inferred_concepts": List of additional relevant concepts the query implies but doesn't explicitly mention
5. "query_type": One of: causal, comparison, solution, definition, general
6. "emphasis_direction": One of: cause, effect, neutral
7. "comparison_pairs": List of [concept1, concept2] pairs if this is a comparison query
8. "highlight_paths": List of [source, target] concept pairs that should be highlighted in the graph
9. "reasoning_chain": List of strings explaining your analysis steps
10. "new_concepts": List of objects with "name" (snake_case), "type" (material/property/phenomenon/process/method/parameter), "definition", "synonyms" (list), "relate_to" (list of [target_concept, relationship_type, confidence])
11. "new_relationships": List of [source, relationship_type, target, confidence] for relationships between EXISTING concepts that should be added

Available relationship types: causes, influences, depends_on, constrains, stabilizes, reduces, improves, enables, detects, measures, observes, processes, forms, transitions_to, replaces, part_of, has_part

Be thorough and scientifically accurate. For new_concepts, only propose concepts that are genuinely novel to the ontology and relevant to the query."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                st.warning("openai package not installed. pip install openai")
                return None
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key) and self._get_client() is not None

    @timed
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        client = self._get_client()
        if client is None:
            return FallbackAnalyzer().analyze_query(query, ontology)

        # Provide ontology context
        concept_list = list(ontology.concepts.keys())
        concept_summary = ", ".join(concept_list[:50])
        if len(concept_list) > 50:
            concept_summary += f" ... and {len(concept_list) - 50} more"

        user_prompt = f"""Analyze this SIB query: "{query}"

Available ontology concepts: {concept_summary}

Return ONLY valid JSON matching the schema described."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            return self._parse_llm_response(parsed, query, ontology)

        except Exception as e:
            st.warning(f"OpenAI analysis failed ({e}), falling back to rule-based")
            return FallbackAnalyzer().analyze_query(query, ontology)

    def _parse_llm_response(self, parsed: Dict, query: str,
                            ontology: DomainOntology) -> QueryAnalysisResult:
        problem_map = {p.value: p for p in SIBCoreProblem}
        primary = problem_map.get(parsed.get("primary_problem", "general"), SIBCoreProblem.GENERAL)
        secondary = [problem_map.get(p, SIBCoreProblem.GENERAL)
                     for p in parsed.get("secondary_problems", [])]

        # Filter concepts to those in ontology
        explicitly_mentioned = [c for c in parsed.get("explicitly_mentioned", [])
                                if c in ontology.concepts]
        inferred = [c for c in parsed.get("inferred_concepts", [])
                    if c in ontology.concepts and c not in explicitly_mentioned]
        all_relevant = list(dict.fromkeys(explicitly_mentioned + inferred))

        # Build priorities
        priorities = {}
        problem_def = SIB_PROBLEM_DEFINITIONS.get(primary, SIB_PROBLEM_DEFINITIONS[SIBCoreProblem.GENERAL])
        problem_concept_set = problem_def.get_ontology_concepts()

        for concept in all_relevant:
            is_explicit = concept in explicitly_mentioned
            priorities[concept] = ConceptPriority(
                concept_name=concept,
                concept_type=ontology.get_concept_type(concept),
                composite_score=0.9 if is_explicit else 0.6,
                direct_score=1.0 if is_explicit else 0.5,
                problem_affinity_score=1.0 if concept in problem_concept_set else 0.5,
                causal_path_score=0.7,
                centrality_bonus=0.0,
                cooccurrence_bonus=0.0,
                is_explicitly_mentioned=is_explicit,
                is_inferred=not is_explicit,
                inference_reason="llm_inferred" if not is_explicit else "llm_explicit"
            )

        comparison_pairs = []
        for pair in parsed.get("comparison_pairs", []):
            if len(pair) == 2:
                comparison_pairs.append(tuple(pair))

        highlight_paths = []
        for path in parsed.get("highlight_paths", []):
            if len(path) == 2:
                resolved_s = ontology.resolve_concept(path[0])
                resolved_t = ontology.resolve_concept(path[1])
                if resolved_s and resolved_t:
                    highlight_paths.append([resolved_s, resolved_t])

        # Store new concepts/relationships for ontology expansion
        self._pending_new_concepts = parsed.get("new_concepts", [])
        self._pending_new_relationships = parsed.get("new_relationships", [])

        reasoning = parsed.get("reasoning_chain", ["LLM analysis completed"])

        return QueryAnalysisResult(
            original_query=query,
            normalized_query=query.lower().strip(),
            primary_problem=primary,
            secondary_problems=secondary,
            problem_confidences={},
            explicitly_mentioned=explicitly_mentioned,
            inferred_concepts=inferred,
            all_relevant_concepts=all_relevant,
            concept_priorities=priorities,
            query_type=parsed.get("query_type", "general"),
            emphasis_direction=parsed.get("emphasis_direction", "cause"),
            comparison_pairs=comparison_pairs,
            subgraph_depth=2,
            priority_threshold=0.3,
            focus_nodes=explicitly_mentioned[:5],
            bridge_nodes=inferred[:3],
            suggested_layout="bisected" if comparison_pairs else "force",
            highlight_paths=highlight_paths,
            visualization_focus=problem_def.visualization_focus,
            reasoning_chain=reasoning,
            confidence=0.85,
        )

class LocalLLMQueryAnalyzer(LLMQueryAnalyzer):
    """Uses a local HuggingFace model for query analysis (no API key needed)."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1500,
                temperature=0.1,
            )
            self._loaded = True
        except Exception as e:
            st.warning(f"Failed to load local model: {e}")
            self._loaded = False

    def is_available(self) -> bool:
        self._load_model()
        return self._loaded

    @timed
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        if not self.is_available():
            return FallbackAnalyzer().analyze_query(query, ontology)

        prompt = f"""[INST] You are an SIB expert. Analyze: "{query}"
Return JSON with: primary_problem (anode_bottleneck|cathode_instability|sei_chemistry|solid_state_interface|low_energy_density|moisture_manufacturing|general), explicitly_mentioned (list of snake_case concept names), inferred_concepts (list), query_type (causal|comparison|solution|definition|general), new_concepts (list of objects with name, type, definition, synonyms, relate_to), new_relationships (list of [source, rel_type, target, confidence]).
Only valid JSON. [/INST]"""

        try:
            result = self._pipeline(prompt)[0]["generated_text"]
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Delegate to OpenAI parser (same format)
                fake_openai = OpenAIQueryAnalyzer()
                return fake_openai._parse_llm_response(parsed, query, ontology)
        except Exception as e:
            st.warning(f"Local LLM parsing failed: {e}")

        return FallbackAnalyzer().analyze_query(query, ontology)

# ============================================================================
# LLM QUERY ANALYZER FACTORY
# ============================================================================
class LLMQueryAnalyzerFactory:
    def __init__(self) -> None:
        self._openai_cache: Optional[OpenAIQueryAnalyzer] = None
        self._local_cache: Optional[LocalLLMQueryAnalyzer] = None
        self._fallback = FallbackAnalyzer()

    def get_analyzer(self, mode: str = "auto",
                     api_key: str = None,
                     local_model: str = None) -> LLMQueryAnalyzer:
        if mode == "openai":
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            return self._openai_cache
        elif mode == "local":
            if self._local_cache is None:
                model = local_model or "mistralai/Mistral-7B-Instruct-v0.2"
                self._local_cache = LocalLLMQueryAnalyzer(model)
            return self._local_cache
        elif mode == "fallback":
            return self._fallback
        else:  # auto
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            if self._openai_cache.is_available():
                return self._openai_cache
            if self._local_cache is None:
                model = local_model or "mistralai/Mistral-7B-Instruct-v0.2"
                self._local_cache = LocalLLMQueryAnalyzer(model)
            if self._local_cache.is_available():
                return self._local_cache
            return self._fallback

# ============================================================================
# ★★★ DYNAMIC ONTOLOGY EXPANDER ★★★
# ============================================================================
class DynamicOntologyExpander:
    REL_STR_TO_ENUM: Dict[str, RelationshipType] = {r.value: r for r in RelationshipType}
    for _k, _v in list(REL_STR_TO_ENUM.items()):
        REL_STR_TO_ENUM[_k.upper()] = _v

    TYPE_STR_TO_ENUM: Dict[str, ConceptType] = {t.value: t for t in ConceptType}

    def __init__(self, ontology: DomainOntology) -> None:
        self.ontology = ontology
        self.mutation_log: List[Dict[str, Any]] = []
        self.session_concepts_added: Set[str] = set()
        self.session_relationships_added: List[Tuple[str, str, RelationshipType, float]] = []
        self.query_bridge_concepts: Dict[str, str] = {}
        self.priority_overrides: Dict[str, float] = {}
        self._base_concept_count: int = len(ontology.concepts)
        self._base_rel_count: int = len(ontology.relationships)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "base_concepts": self._base_concept_count,
            "base_relationships": self._base_rel_count,
            "concepts_added": len(self.session_concepts_added),
            "relationships_added": len(self.session_relationships_added),
            "bridge_concepts": len(self.query_bridge_concepts),
            "priority_overrides": len(self.priority_overrides),
            "total_mutations": len(self.mutation_log),
        }

    def apply_query_analysis(self, analysis: QueryAnalysisResult,
                             analyzer: LLMQueryAnalyzer = None) -> Dict[str, Any]:
        changes = {"concepts_added": [], "relationships_added": [], "bridges_created": []}

        # 1. Priority overrides
        for concept_name, priority in analysis.concept_priorities.items():
            if concept_name in self.ontology.concepts:
                self.priority_overrides[concept_name] = priority.composite_score

        # 2. Extract new concepts & relationships from LLM
        new_concepts_raw = []
        new_rels_raw = []
        if isinstance(analyzer, OpenAIQueryAnalyzer):
            new_concepts_raw = getattr(analyzer, '_pending_new_concepts', [])
            new_rels_raw = getattr(analyzer, '_pending_new_relationships', [])
        elif isinstance(analyzer, LocalLLMQueryAnalyzer):
            new_concepts_raw = getattr(analyzer, '_pending_new_concepts', [])
            new_rels_raw = getattr(analyzer, '_pending_new_relationships', [])

        for concept_data in new_concepts_raw:
            result = self._add_concept_from_llm(concept_data, analysis.original_query)
            if result:
                changes["concepts_added"].append(result)

        for rel_data in new_rels_raw:
            result = self._add_relationship_from_llm(rel_data, analysis.original_query)
            if result:
                changes["relationships_added"].append(result)

        # 3. Bridge concepts for inferred but missing
        for concept in analysis.inferred_concepts:
            if concept not in self.ontology.concepts:
                bridge_result = self._create_bridge_concept(
                    concept, analysis.original_query, analysis.primary_problem
                )
                if bridge_result:
                    changes["bridges_created"].append(bridge_result)

        self.ontology._build_synonym_index()
        return changes

    def _add_concept_from_llm(self, concept_data: Dict, source_query: str) -> Optional[Dict]:
        name = concept_data.get("name", "").strip().lower().replace(" ", "_")
        if not name or name in self.ontology.concepts or name in self.session_concepts_added:
            return None
        type_str = concept_data.get("type", "general")
        concept_type = self.TYPE_STR_TO_ENUM.get(type_str, ConceptType.GENERAL)
        synonyms = set()
        for syn in concept_data.get("synonyms", []):
            if isinstance(syn, str):
                synonyms.add(syn.lower().strip())
        definition = concept_data.get("definition", f"LLM-inferred concept from query: {source_query}")
        self.ontology._add_concept(name, concept_type, synonyms=synonyms, definition=definition)
        self.ontology.synonym_to_canonical[name.lower()] = name
        for syn in synonyms:
            self.ontology.synonym_to_canonical[syn] = name
        self.session_concepts_added.add(name)
        for rel_tuple in concept_data.get("relate_to", []):
            if len(rel_tuple) >= 2:
                target = rel_tuple[0]
                rel_type_str = rel_tuple[1] if len(rel_tuple) > 1 else "influences"
                conf = rel_tuple[2] if len(rel_tuple) > 2 else 0.7
                rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
                if target in self.ontology.concepts:
                    self.ontology._add_relationship(name, rel_enum, target, float(conf))
                    self.session_relationships_added.append((name, target, rel_enum, float(conf)))
        mutation = {"type":"add_concept","concept":name,"concept_type":concept_type.value,
                    "synonyms":list(synonyms),"definition":definition,
                    "source_query":source_query,"timestamp":datetime.now().isoformat()}
        self.mutation_log.append(mutation)
        return {"name":name, "type":concept_type.value, "synonyms":list(synonyms)}

    def _add_relationship_from_llm(self, rel_data: List, source_query: str) -> Optional[Dict]:
        if len(rel_data) < 3:
            return None
        source = str(rel_data[0]).strip().lower().replace(" ", "_")
        rel_type_str = str(rel_data[1]).upper()
        target = str(rel_data[2]).strip().lower().replace(" ", "_")
        confidence = float(rel_data[3]) if len(rel_data) > 3 else 0.7
        if source not in self.ontology.concepts or target not in self.ontology.concepts:
            return None
        for existing in self.ontology.relationships:
            if (existing.source == source and existing.target == target and
                    existing.rel_type.value == rel_type_str.lower()):
                return None
        rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
        self.ontology._add_relationship(source, rel_enum, target, confidence)
        self.session_relationships_added.append((source, target, rel_enum, confidence))
        mutation = {"type":"add_relationship","source":source,"target":target,
                    "rel_type":rel_enum.value,"confidence":confidence,
                    "source_query":source_query,"timestamp":datetime.now().isoformat()}
        self.mutation_log.append(mutation)
        return {"source":source, "target":target, "rel_type":rel_enum.value, "confidence":confidence}

    def _create_bridge_concept(self, missing_concept: str, source_query: str,
                                problem: SIBCoreProblem) -> Optional[Dict]:
        bridge_name = f"query_bridge_{missing_concept.replace(' ', '_').lower()}"
        if bridge_name in self.ontology.concepts:
            return None
        problem_def = SIB_PROBLEM_DEFINITIONS.get(problem, SIB_PROBLEM_DEFINITIONS[SIBCoreProblem.GENERAL])
        self.ontology._add_concept(
            bridge_name, ConceptType.GENERAL,
            synonyms={missing_concept.lower()},
            definition=f"Query-inferred bridge concept: '{missing_concept}' (from query about {problem.value})"
        )
        self.ontology.synonym_to_canonical[bridge_name] = bridge_name
        self.ontology.synonym_to_canonical[missing_concept.lower()] = bridge_name
        connected = []
        for key_concept in problem_def.key_concepts[:3]:
            if key_concept in self.ontology.concepts:
                self.ontology._add_relationship(bridge_name, RelationshipType.BRIDGE, key_concept, 0.5)
                self.session_relationships_added.append((bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                connected.append(key_concept)
        self.session_concepts_added.add(bridge_name)
        self.query_bridge_concepts[bridge_name] = source_query
        mutation = {"type":"create_bridge","bridge_name":bridge_name,"original_term":missing_concept,
                    "connected_to":connected,"source_query":source_query,"timestamp":datetime.now().isoformat()}
        self.mutation_log.append(mutation)
        return {"bridge":bridge_name, "for":missing_concept, "connected_to":connected}

    def get_priority_boosted_scores(self, base_priorities: Dict[str, ConceptPriority]
                                    ) -> Dict[str, ConceptPriority]:
        boosted = {}
        for name, priority in base_priorities.items():
            boost = self.priority_overrides.get(name, 0.0)
            if boost > 0:
                boosted_priority = copy.deepcopy(priority)
                boosted_priority.composite_score = min(boosted_priority.composite_score + boost * 0.2, 1.0)
                boosted_priority.centrality_bonus = boost * 0.2
                boosted[name] = boosted_priority
            else:
                boosted[name] = priority
        return boosted

    def undo_last_mutation(self) -> Optional[Dict]:
        if not self.mutation_log:
            return None
        mutation = self.mutation_log.pop()
        mut_type = mutation["type"]
        if mut_type == "add_concept":
            name = mutation["concept"]
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
                self.session_concepts_added.discard(name)
                self.ontology.relationships = [r for r in self.ontology.relationships if r.source != name and r.target != name]
                self.session_relationships_added = [(s,t,rt,c) for s,t,rt,c in self.session_relationships_added if s != name and t != name]
            self.ontology._build_synonym_index()
        elif mut_type == "add_relationship":
            source = mutation["source"]
            target = mutation["target"]
            rel_type = self.REL_STR_TO_ENUM.get(mutation["rel_type"], RelationshipType.INFLUENCES)
            self.ontology.relationships = [r for r in self.ontology.relationships if not (r.source == source and r.target == target and r.rel_type == rel_type)]
            self.session_relationships_added = [(s,t,rt,c) for s,t,rt,c in self.session_relationships_added if not (s == source and t == target and rt == rel_type)]
        elif mut_type == "create_bridge":
            bridge_name = mutation["bridge_name"]
            if bridge_name in self.ontology.concepts:
                del self.ontology.concepts[bridge_name]
                self.session_concepts_added.discard(bridge_name)
                self.query_bridge_concepts.pop(bridge_name, None)
                self.ontology.relationships = [r for r in self.ontology.relationships if r.source != bridge_name and r.target != bridge_name]
            self.ontology._build_synonym_index()
        return mutation

    def reset_to_base(self) -> Dict[str, int]:
        removed_concepts = 0
        removed_rels = 0
        for name in list(self.session_concepts_added):
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
                removed_concepts += 1
        base_rels = self.ontology.relationships[:self._base_rel_count]
        removed_rels = len(self.ontology.relationships) - len(base_rels)
        self.ontology.relationships = base_rels
        self.session_concepts_added.clear()
        self.session_relationships_added.clear()
        self.query_bridge_concepts.clear()
        self.priority_overrides.clear()
        self.mutation_log.clear()
        self.ontology._build_synonym_index()
        return {"concepts_removed": removed_concepts, "relationships_removed": removed_rels}

    def export_mutations(self) -> Dict[str, Any]:
        return {
            "stats": self.stats,
            "mutations": self.mutation_log,
            "concepts_added": list(self.session_concepts_added),
            "bridge_concepts": dict(self.query_bridge_concepts),
            "priority_overrides": {k: round(v, 3) for k, v in self.priority_overrides.items()},
        }

    def import_mutations(self, data: Dict[str, Any]) -> int:
        count = 0
        for mutation in data.get("mutations", []):
            if mutation["type"] == "add_concept":
                result = self._add_concept_from_llm(
                    {"name": mutation["concept"],
                     "type": mutation.get("concept_type", "general"),
                     "synonyms": mutation.get("synonyms", []),
                     "definition": mutation.get("definition", "")},
                    mutation.get("source_query", "restored")
                )
                if result:
                    count += 1
            elif mutation["type"] == "add_relationship":
                result = self._add_relationship_from_llm(
                    [mutation["source"], mutation["rel_type"],
                     mutation["target"], mutation.get("confidence", 0.7)],
                    mutation.get("source_query", "restored")
                )
                if result:
                    count += 1
        self.ontology._build_synonym_index()
        return count

# ============================================================================
# PRIORITY-GUIDED SUBGRAPH EXTRACTOR
# ============================================================================
class PriorityGuidedSubgraphExtractor:
    def __init__(self, full_graph: nx.DiGraph,
                 ontology: DomainOntology,
                 expander: DynamicOntologyExpander) -> None:
        self.full_graph = full_graph
        self.ontology = ontology
        self.expander = expander

    def extract(self, analysis: QueryAnalysisResult) -> nx.DiGraph:
        boosted = self.expander.get_priority_boosted_scores(analysis.concept_priorities)
        concepts_above = analysis.get_concepts_above_threshold()
        seed_nodes = set(analysis.focus_nodes + concepts_above)
        visited = set(seed_nodes)
        frontier = deque(seed_nodes)
        depth_map = {n: 0 for n in seed_nodes}
        max_depth = analysis.subgraph_depth

        while frontier:
            current = frontier.popleft()
            current_depth = depth_map[current]
            if current_depth >= max_depth:
                continue
            for neighbor in list(self.full_graph.predecessors(current)) + list(self.full_graph.successors(current)):
                if neighbor in visited:
                    continue
                include = False
                if neighbor in boosted:
                    if (boosted[neighbor].composite_score >= analysis.priority_threshold * 0.5
                            or current_depth < max_depth - 1):
                        include = True
                elif neighbor in self.expander.session_concepts_added:
                    include = True
                elif neighbor in self.expander.query_bridge_concepts:
                    include = True
                else:
                    if current_depth < max_depth - 1:
                        include = True
                if include:
                    visited.add(neighbor)
                    depth_map[neighbor] = current_depth + 1
                    frontier.append(neighbor)

        subgraph = nx.DiGraph()
        for node in visited:
            if node in self.full_graph.nodes:
                subgraph.add_node(node, **self.full_graph.nodes[node])
                if node in boosted:
                    subgraph.nodes[node]["priority_score"] = boosted[node].composite_score
                    subgraph.nodes[node]["is_explicit"] = boosted[node].is_explicitly_mentioned
                    subgraph.nodes[node]["is_inferred"] = boosted[node].is_inferred
                elif node in self.expander.session_concepts_added:
                    subgraph.nodes[node]["priority_score"] = 0.5
                    subgraph.nodes[node]["is_explicit"] = False
                    subgraph.nodes[node]["is_inferred"] = True
                    subgraph.nodes[node]["is_llm_added"] = True
                else:
                    subgraph.nodes[node]["priority_score"] = 0.2

        for u, v, attrs in self.full_graph.edges(data=True):
            if u in subgraph and v in subgraph:
                edge_attrs = dict(attrs)
                for path in analysis.highlight_paths:
                    if len(path) >= 2:
                        for i in range(len(path) - 1):
                            if (path[i] == u and path[i+1] == v) or (path[i] == v and path[i+1] == u):
                                edge_attrs["highlighted"] = True
                                edge_attrs["width"] = max(edge_attrs.get("width", 1.0), 4.0)
                subgraph.add_edge(u, v, **edge_attrs)

        subgraph.graph["query_analysis"] = analysis
        subgraph.graph["suggested_layout"] = analysis.suggested_layout
        subgraph.graph["emphasis_direction"] = analysis.emphasis_direction
        return subgraph

# ============================================================================
# QUERY-DRIVEN VISUALIZER (PyVis)
# ============================================================================
class QueryDrivenVisualizer:
    def __init__(self, ontology: DomainOntology) -> None:
        self.ontology = ontology
        self.type_colors = {
            ConceptType.MATERIAL.value: "#FF6B6B",
            ConceptType.PROPERTY.value: "#4ECDC4",
            ConceptType.PHENOMENON.value: "#FFE66D",
            ConceptType.METHOD.value: "#95E1D3",
            ConceptType.PARAMETER.value: "#F38181",
            ConceptType.PROCESS.value: "#AA96DA",
            ConceptType.MODEL.value: "#FCBAD3",
            ConceptType.GENERAL.value: "#A8D8EA",
        }

    def render_pyvis(self, subgraph: nx.DiGraph, analysis: QueryAnalysisResult,
                     height: str = "700px") -> str:
        net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
        if analysis.suggested_layout == "bisected":
            net.barnes_hut(gravity=80, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.95)
        else:
            net.barnes_hut(gravity=50, central_gravity=0.2, spring_length=150, spring_strength=0.04, damping=0.9)

        for node, attrs in subgraph.nodes(data=True):
            concept_type = attrs.get("concept_type", "general")
            priority = attrs.get("priority_score", 0.2)
            is_explicit = attrs.get("is_explicit", False)
            is_llm_added = attrs.get("is_llm_added", False)
            size = 15 + priority * 35
            color = self.type_colors.get(concept_type, "#A8D8EA")
            if is_explicit:
                border_width, border_color, shape = 4, "#FF0000", "dot"
            elif is_llm_added:
                border_width, border_color, shape = 3, "#00FF00", "diamond"
            else:
                border_width, border_color, shape = 1, "#666666", "dot"
            title = f"<b>{attrs.get('hierarchy_label', node)}</b><br>Type: {concept_type}<br>Priority: {priority:.2f}"
            if is_llm_added:
                title += "<br>⚠️ LLM-inferred concept"
            defn = attrs.get("definition", "")
            if defn:
                title += f"<br><i>{defn[:150]}...</i>"
            net.add_node(node, label=attrs.get("hierarchy_label", node).replace("_", " ").split(" › ")[-1],
                         size=size, color=color, border_width=border_width, border_color=border_color,
                         shape=shape, title=title, font={"size": 10 + priority * 6})

        for u, v, attrs in subgraph.edges(data=True):
            color = attrs.get("color", "#888888")
            width = attrs.get("width", 1.0)
            style = attrs.get("style", "solid")
            highlighted = attrs.get("highlighted", False)
            if highlighted:
                color = "#FF0000"
                width = max(width, 4.0)
            dashes = style == "dashed" or attrs.get("source_type") == "cooccurrence"
            net.add_edge(u, v, color=color, width=width, dashes=dashes,
                         title=f"{u} → {v}<br>Type: {attrs.get('edge_type','unknown')}",
                         arrows="to", arrowStrikethrough=False)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            net.save_graph(f.name)
            html_content = Path(f.name).read_text(encoding='utf-8')
        return html_content

    def render_comparison_pyvis(self, subgraph: nx.DiGraph, analysis: QueryAnalysisResult,
                                 height: str = "700px") -> str:
        if not analysis.comparison_pairs:
            return self.render_pyvis(subgraph, analysis, height)
        net = Network(height=height, width="100%", directed=True, notebook=False)
        left_nodes, right_nodes = set(), set()
        if analysis.comparison_pairs:
            pair = analysis.comparison_pairs[0]
            left_term, right_term = pair[0].lower(), pair[1].lower()
            for node in subgraph.nodes:
                node_lower = node.lower()
                if left_term in node_lower:
                    left_nodes.add(node)
                elif right_term in node_lower:
                    right_nodes.add(node)
        left_list, right_list = list(left_nodes), list(right_nodes)
        other_nodes = [n for n in subgraph.nodes if n not in left_nodes and n not in right_nodes]
        for i, node in enumerate(left_list):
            subgraph.nodes[node]["x"] = -300
            subgraph.nodes[node]["y"] = (i - len(left_list)/2) * 80
            subgraph.nodes[node]["fixed"] = True
        for i, node in enumerate(right_list):
            subgraph.nodes[node]["x"] = 300
            subgraph.nodes[node]["y"] = (i - len(right_list)/2) * 80
            subgraph.nodes[node]["fixed"] = True
        for i, node in enumerate(other_nodes):
            subgraph.nodes[node]["x"] = 0
            subgraph.nodes[node]["y"] = (i - len(other_nodes)/2) * 60
            subgraph.nodes[node]["fixed"] = True
        net.barnes_hut(enabled=False)
        for node, attrs in subgraph.nodes(data=True):
            color = "#4ECDC4" if node in left_nodes else "#FF6B6B" if node in right_nodes else "#AAAAAA"
            priority = attrs.get("priority_score", 0.2)
            size = 15 + priority * 30
            net.add_node(node, label=attrs.get("hierarchy_label", node).split(" › ")[-1],
                         size=size, color=color, x=attrs.get("x",0), y=attrs.get("y",0),
                         fixed=attrs.get("fixed", False), title=f"<b>{node}</b><br>{attrs.get('definition','')[:100]}",
                         font={"size":11})
        for u, v, attrs in subgraph.edges(data=True):
            net.add_edge(u, v, color=attrs.get("color","#888888"), width=attrs.get("width",1.0), arrows="to")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            net.save_graph(f.name)
            html_content = Path(f.name).read_text(encoding='utf-8')
        return html_content

# ============================================================================
# SESSION STATE MANAGER FOR LLM QUERY INTEGRATION
# ============================================================================
class QuerySessionManager:
    SESSION_KEY = "sib_query_session"

    @classmethod
    def init_session(cls) -> Dict[str, Any]:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {
                "query_history": [], "analysis_history": [], "mutation_history": [],
                "analyzer_mode": "auto", "total_concepts_added": 0,
                "total_relationships_added": 0,
            }
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def record_query(cls, query: str, analysis: QueryAnalysisResult,
                     mutations: Dict[str, Any]) -> None:
        session = cls.init_session()
        session["query_history"].append(query)
        session["analysis_history"].append({
            "query": query, "primary_problem": analysis.primary_problem.value,
            "query_type": analysis.query_type,
            "concepts_found": len(analysis.all_relevant_concepts),
            "explicit": len(analysis.explicitly_mentioned),
            "inferred": len(analysis.inferred_concepts),
            "confidence": analysis.confidence,
            "timestamp": datetime.now().isoformat(),
        })
        session["mutation_history"].append({
            "query": query,
            "concepts_added": len(mutations.get("concepts_added", [])),
            "relationships_added": len(mutations.get("relationships_added", [])),
            "bridges_created": len(mutations.get("bridges_created", [])),
            "timestamp": datetime.now().isoformat(),
        })
        session["total_concepts_added"] += len(mutations.get("concepts_added", []))
        session["total_relationships_added"] += len(mutations.get("relationships_added", []))

    @classmethod
    def get_session(cls) -> Dict[str, Any]:
        return cls.init_session()

    @classmethod
    def clear_session(cls) -> None:
        if cls.SESSION_KEY in st.session_state:
            del st.session_state[cls.SESSION_KEY]

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
# EXISTING VISUALIZATION FUNCTIONS (from v6.1) – included for completeness
# ============================================================================
def render_pyvis_graph(nx_graph, concept_abstract_map, physics_enabled=True,
                       cmap_name="viridis", top_n_nodes=0, theme=None,
                       physics_preset=None, show_edge_weights=False,
                       edge_label_mode="hover", **kwargs):
    if theme is None:
        theme = {"bg":"#ffffff","font":"#1e293b","node_border":"#f8fafc",
                 "highlight_bg":"#ff6b6b","hover_bg":"#ffd93d","shadow_color":"rgba(0,0,0,0.15)",
                 "tooltip_bg":"rgba(255,255,255,0.95)","tooltip_text":"#1e293b","tooltip_border":"#cbd5e1"}
    if physics_preset is None:
        physics_preset = {"damping":0.55,"gravity":-2500,"spring_length":140,
                          "spring_strength":0.05,"central_gravity":0.25,"stabilization":2500}
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

def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except:
        return ["#ff6b6b"]*n

# ============================================================================
# THEME PRESETS & SIDEBAR
# ============================================================================
THEME_PRESETS = {
    "Bright (Default)": {"bg":"#ffffff","font":"#1e293b","tooltip_bg":"rgba(255,255,255,0.95)",
                         "tooltip_border":"#cbd5e1","tooltip_text":"#1e293b",
                         "edge_unknown":"rgba(148,163,184,0.3)","node_border":"#f8fafc",
                         "highlight_bg":"#ff6b6b","hover_bg":"#ffd93d","shadow_color":"rgba(0,0,0,0.15)",
                         "plotly_bg":"#ffffff","plotly_paper":"#ffffff","grid_color":"#e2e8f0","axis_color":"#64748b"},
    "Dark": {"bg":"#0f172a","font":"#e2e8f0","tooltip_bg":"rgba(15,23,42,0.95)",
             "tooltip_border":"#334155","tooltip_text":"#e2e8f0",
             "edge_unknown":"rgba(148,163,184,0.4)","node_border":"#f8fafc",
             "highlight_bg":"#ff6b6b","hover_bg":"#ffd93d","shadow_color":"rgba(0,0,0,0.6)",
             "plotly_bg":"#0f172a","plotly_paper":"#0f172a","grid_color":"#1e293b","axis_color":"#94a3b8"},
}
SUPPORTED_COLORMAPS = {"viridis":"Viridis","plasma":"Plasma","inferno":"Inferno","magma":"Magma",
                       "cividis":"Cividis","turbo":"Turbo","jet":"Jet","rainbow":"Rainbow",
                       "coolwarm":"Coolwarm","RdBu":"RdBu","Spectral":"Spectral",
                       "tab10":"Set1","tab20":"Set2"}

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
# UI FUNCTIONS FOR LLM QUERY PANEL
# ============================================================================
def render_llm_query_panel(ontology: DomainOntology,
                           expander: DynamicOntologyExpander,
                           full_graph: nx.DiGraph) -> Optional[QueryAnalysisResult]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 LLM-Guided Query")
    st.sidebar.caption("Ask a question to dynamically expand the ontology and focus the graph")

    session = QuerySessionManager.get_session()
    mode = st.sidebar.selectbox(
        "Analysis Engine",
        ["auto", "fallback", "openai", "local"],
        index=["auto", "fallback", "openai", "local"].index(session.get("analyzer_mode", "auto")),
        key="llm_mode_select"
    )
    session["analyzer_mode"] = mode

    api_key = None
    if mode in ("auto", "openai"):
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password",
                                        value=os.environ.get("OPENAI_API_KEY", ""),
                                        key="openai_key_input")
    local_model = None
    if mode in ("auto", "local"):
        local_model = st.sidebar.text_input("Local Model (optional)",
                                            value="mistralai/Mistral-7B-Instruct-v0.2",
                                            key="local_model_input")

    st.sidebar.markdown("**Example queries:**")
    example_queries = []
    for problem, pdef in SIB_PROBLEM_DEFINITIONS.items():
        if pdef.example_queries:
            example_queries.extend(pdef.example_queries[:1])
    selected_example = st.sidebar.selectbox("Or select an example:", [""] + example_queries,
                                            key="example_query_select")
    query = st.sidebar.text_area("Your SIB question:", value=selected_example, height=100,
                                 key="llm_query_input",
                                 placeholder="e.g., Why can't sodium intercalate into graphite like lithium does?")
    submitted = st.sidebar.button("🚀 Analyze & Expand Ontology", type="primary", key="llm_submit")
    if not submitted or not query.strip():
        return None

    factory = LLMQueryAnalyzerFactory()
    analyzer = factory.get_analyzer(mode=mode, api_key=api_key, local_model=local_model)

    if isinstance(analyzer, OpenAIQueryAnalyzer):
        st.sidebar.info("🤖 Using **OpenAI GPT-4o-mini**")
    elif isinstance(analyzer, LocalLLMQueryAnalyzer):
        st.sidebar.info("🖥️ Using **Local LLM**")
    else:
        st.sidebar.info("📋 Using **Rule-based fallback**")

    with st.sidebar.spinner("Analyzing query..."):
        analysis = analyzer.analyze_query(query, ontology)
    with st.sidebar.spinner("Expanding ontology..."):
        mutations = expander.apply_query_analysis(analysis, analyzer)

    QuerySessionManager.record_query(query, analysis, mutations)

    st.sidebar.success(f"✅ Analysis complete (confidence: {analysis.confidence:.0%})")
    st.sidebar.caption(f"Primary problem: **{analysis.primary_problem.value}**")
    st.sidebar.caption(f"Explicit concepts: {len(analysis.explicitly_mentioned)} | "
                       f"Inferred: {len(analysis.inferred_concepts)}")
    if mutations["concepts_added"]:
        st.sidebar.warning(f"🆕 {len(mutations['concepts_added'])} new concept(s) added to ontology")
        for c in mutations["concepts_added"]:
            st.sidebar.markdown(f"  - `{c['name']}` ({c['type']})")
    if mutations["bridges_created"]:
        st.sidebar.info(f"🌉 {len(mutations['bridges_created'])} bridge concept(s) created")
        for b in mutations["bridges_created"]:
            st.sidebar.markdown(f"  - `{b['bridge']}` ← `{b['for']}`")
    return analysis

def render_mutation_controls(expander: DynamicOntologyExpander) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Ontology Mutations")
    stats = expander.stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Concepts +", stats["concepts_added"])
    col2.metric("Relations +", stats["relationships_added"])
    if stats["total_mutations"] > 0:
        with st.sidebar.expander("📋 Mutation Log", expanded=False):
            for i, mut in enumerate(expander.mutation_log[-10:], 1):
                if mut["type"] == "add_concept":
                    st.markdown(f"{i}. ➕ `{mut['concept']}`")
                elif mut["type"] == "add_relationship":
                    st.markdown(f"{i}. 🔗 `{mut['source']}` → `{mut['target']}`")
                elif mut["type"] == "create_bridge":
                    st.markdown(f"{i}. 🌉 `{mut['bridge_name']}`")
        col_undo, col_reset = st.sidebar.columns(2)
        if col_undo.button("↩️ Undo Last", key="undo_mutation"):
            undone = expander.undo_last_mutation()
            if undone:
                st.sidebar.toast(f"Undone: {undone['type']}")
                st.rerun()
        if col_reset.button("🔄 Reset All", key="reset_mutations"):
            result = expander.reset_to_base()
            st.sidebar.toast(f"Reset: {result['concepts_removed']} concepts, "
                             f"{result['relationships_removed']} relations removed")
            st.rerun()
        if st.sidebar.button("📦 Export Mutations", key="export_mutations"):
            exported = expander.export_mutations()
            st.sidebar.download_button(
                "Download JSON",
                data=json.dumps(exported, indent=2),
                file_name=f"sib_ontology_mutations_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
                key="download_mutations"
            )

def render_query_history() -> None:
    session = QuerySessionManager.get_session()
    if not session["query_history"]:
        return
    st.sidebar.markdown("---")
    with st.sidebar.expander("📜 Query History", expanded=False):
        for i, entry in enumerate(reversed(session["analysis_history"][-10:]), 1):
            st.markdown(f"**{i}.** {entry['query'][:60]}...")
            st.caption(f"  Problem: {entry['primary_problem']} | "
                       f"Type: {entry['query_type']} | "
                       f"Concepts: {entry['concepts_found']}")
            st.caption(f"  Added: +{session['mutation_history'][-i].get('concepts_added', 0)} concepts, "
                       f"+{session['mutation_history'][-i].get('relationships_added', 0)} rels")

def render_analysis_details(analysis: QueryAnalysisResult) -> None:
    st.markdown("## 📊 Query Analysis Results")
    with st.expander("🧠 Reasoning Chain", expanded=True):
        for step in analysis.reasoning_chain:
            st.markdown(f"→ {step}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Problem", analysis.primary_problem.value.replace("_", " "))
    col2.metric("Query Type", analysis.query_type)
    col3.metric("Confidence", f"{analysis.confidence:.0%}")
    if analysis.problem_confidences:
        st.markdown("### Problem Affinity Scores")
        probs_df = pd.DataFrame([
            {"Problem": k.replace("_", " ").title(), "Score": v}
            for k, v in sorted(analysis.problem_confidences.items(),
                               key=lambda x: -x[1]) if v > 0
        ])
        if not probs_df.empty:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(probs_df["Problem"], probs_df["Score"], color="#4ECDC4")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Affinity Score")
            st.pyplot(fig)
            plt.close(fig)
    st.markdown("### Concept Priority Rankings")
    top = analysis.get_top_concepts(15)
    if top:
        priority_data = [cp.to_dict() for cp in top]
        df = pd.DataFrame(priority_data)
        def highlight_row(row):
            if row.get("explicit", False):
                return ["background-color: #d4edda"] * len(row)
            elif row.get("inferred", False):
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)
    if analysis.highlight_paths:
        st.markdown("### 🔴 Highlighted Causal Paths")
        for path in analysis.highlight_paths[:5]:
            path_str = " → ".join(f"**{n.replace('_', ' ')}**" for n in path)
            st.markdown(path_str)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🔋 SIB Quantitative Descriptor Graph v6.2 – LLM Query")
    st.caption("Integrated ontology, batch processing, GNN, and LLM-directed dynamic expansion.")

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    # Initialize expander if not present
    if 'expander' not in st.session_state:
        st.session_state.expander = DynamicOntologyExpander(ontology)
    expander = st.session_state.expander

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

    # Build graph (simplified demo; replace with full v6.1 pipeline for real data)
    if st.button("🚀 Build Concept Graph (Demo)", type="primary") or st.session_state.get("batch_trigger"):
        # Create a graph from the ontology
        G = nx.DiGraph()
        for name in ontology.concepts.keys():
            G.add_node(name, concept_type=ontology.get_concept_type(name).value,
                       definition=ontology.get_definition(name))
        for rel in ontology.relationships:
            G.add_edge(rel.source, rel.target, weight=rel.confidence,
                       edge_type=rel.rel_type.value, color=get_edge_color(rel.rel_type),
                       width=get_edge_width(rel.rel_type), style=get_edge_style(rel.rel_type))
        st.session_state.analysis_data = {
            "nx_graph": G,
            "valid_concepts": list(G.nodes()),
            "concept_abstract_map": {n: [1,2,3] for n in G.nodes()},
            "distill_df": pd.DataFrame({"concept": list(G.nodes()), "frequency": [5]*len(G.nodes())}),
            "top_scores": pd.DataFrame({"concept_u": ["hard_carbon"], "concept_v": ["energy_density"], "composite_score": [0.8]}),
            "df_filtered": df, "selected_text_cols": selected_text_cols
        }
        st.success("Graph built (demo). Replace with full v6.1 pipeline for real data.")

    # Render LLM query panel in sidebar
    full_graph = st.session_state.analysis_data.get("nx_graph", nx.DiGraph()) if st.session_state.get("analysis_data") else nx.DiGraph()
    analysis = render_llm_query_panel(ontology, expander, full_graph)
    render_mutation_controls(expander)
    render_query_history()

    # Main display
    if st.session_state.get("analysis_data"):
        data = st.session_state.analysis_data
        G = data["nx_graph"]
        theme = THEME_PRESETS.get(st.session_state.get("theme", "Bright (Default)"), THEME_PRESETS["Bright (Default)"])

        # If we have a query analysis, show the focused subgraph
        if analysis:
            render_analysis_details(analysis)
            extractor = PriorityGuidedSubgraphExtractor(G, ontology, expander)
            subgraph = extractor.extract(analysis)
            visualizer = QueryDrivenVisualizer(ontology)
            if analysis.query_type == "comparison" and analysis.comparison_pairs:
                html = visualizer.render_comparison_pyvis(subgraph, analysis)
            else:
                html = visualizer.render_pyvis(subgraph, analysis)
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            # Show full graph
            tabs = st.tabs(["📊 Graph", "🧪 Analytics", "📥 Export"])
            with tabs[0]:
                st.subheader("Interactive Graph")
                render_pyvis_graph(G, data["concept_abstract_map"], cmap_name=st.session_state.get("cmap_name","viridis"), theme=theme)
            with tabs[1]:
                st.subheader("Distillation & Analytics")
                if "distill_df" in data:
                    st.dataframe(data["distill_df"])
                if "top_scores" in data and not data["top_scores"].empty:
                    st.dataframe(data["top_scores"])
            with tabs[2]:
                st.subheader("Export")
                st.info("Export functionality available in full v6.1.")
    else:
        st.info("Build the concept graph first (click the button above), then use the LLM query panel in the sidebar.")

if __name__ == "__main__":
    main()
