#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIB LLM-DIRECTED CONCEPT GRAPH - COMPLETE STANDALONE VERSION
=============================================================
A complete, runnable script that wraps the SIB Concept Graph with LLM-directed
query analysis for the six core Sodium-Ion Battery problems.

USAGE:
    python sib_llm_directed_graph.py

REQUIREMENTS:
    pip install streamlit networkx numpy matplotlib pyvis plotly torch

For local LLM support (optional):
    pip install transformers sentence-transformers
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from enum import Enum
from collections import defaultdict
from pathlib import Path
import io
import base64
import warnings
import traceback

warnings.filterwarnings('ignore')

# Optional imports
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Warning: pyvis not installed. Interactive graphs disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SIB LLM-Directed Concept Graph",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# ENUMS
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
    SOLUTION = "solution"
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
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES,
              RelationshipType.FORMS, RelationshipType.STABILIZES,
              RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO}
    if rel_type in STRONG:
        return 3.0
    elif rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE}
    return "dashed" if rel_type in DASHED else "solid"


# ============================================================================
# DATA CLASSES
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


@dataclass
class QueryAnalysisResult:
    original_query: str
    primary_problem: SIBCoreProblem
    secondary_problems: List[SIBCoreProblem]
    problem_confidences: Dict[SIBCoreProblem, float]
    matched_concepts: List[str]
    inferred_concepts: List[str]
    all_relevant_concepts: List[str]
    emphasis_keywords: List[str]
    emphasis_direction: str
    subgraph_depth: int
    include_phenomena: bool
    include_solutions: bool
    include_quantities: bool
    reasoning: str
    suggested_visualizations: List[str]
    graph_construction_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# COMPLETE PROBLEM DEFINITIONS (ALL 6 PROBLEMS)
# ============================================================================
SIB_PROBLEM_DEFINITIONS: Dict[SIBCoreProblem, SIBProblemDefinition] = {
    SIBCoreProblem.ANODE_BOTTLENECK: SIBProblemDefinition(
        problem_id=SIBCoreProblem.ANODE_BOTTLENECK,
        title="The Anode Bottleneck: Graphite Incompatibility",
        scientific_description=(
            "Sodium ions (Na⁺, radius ~1.02 Å) cannot effectively intercalate into standard "
            "graphite layers (interlayer spacing ~3.35 Å) due to thermodynamic instability. "
            "Unlike lithium (Li⁺, radius ~0.76 Å), Na forms unstable NaC₆₄ compounds with "
            "graphite rather than the stable LiC₆. This fundamental incompatibility necessitates "
            "entirely different anode architectures for SIBs."
        ),
        root_cause="Na⁺ ionic radius (1.02 Å) is ~34% larger than Li⁺ (0.76 Å), preventing "
                   "stable intercalation into graphite's limited interlayer spacing",
        key_concepts=[
            "hard_carbon", "alloying_anode", "intercalation_anode", "sodium_metal",
            "graphite_incompatibility", "initial_coulombic_efficiency", "ice",
            "sodium_storage_mechanism", "pore_filling_mechanism", "insertion_mechanism",
            "volume_expansion"
        ],
        key_relationships=[
            ("hard_carbon", "INFLUENCES", "specific_capacity"),
            ("hard_carbon", "INFLUENCES", "coulombic_efficiency"),
            ("hard_carbon", "CAUSES", "low_ice"),
            ("alloying_anode", "INFLUENCES", "specific_capacity"),
            ("alloying_anode", "CAUSES", "volume_expansion"),
            ("volume_expansion", "CAUSES", "cycle_life"),
            ("pre_sodiation", "INFLUENCES", "coulombic_efficiency"),
            ("intercalation", "INFLUENCES", "specific_capacity"),
        ],
        solution_directions=[
            "Optimize hard carbon microstructure (closed pore vs. open pore ratio)",
            "Develop alloying anodes with nanostructuring to accommodate volume change",
            "Apply pre-sodiation techniques to improve ICE",
            "Explore conversion anodes (metal sulfides/oxides) for high capacity",
            "Investigate MXene-based anodes for high rate capability"
        ],
        relevant_materials=["hard_carbon", "alloying_anode", "intercalation_anode",
                          "sodium_metal", "mxene", "conversion_anode"],
        relevant_phenomena=["intercalation", "volume_expansion", "sei_formation",
                          "sodium_plating_stripping", "conversion_reaction"],
        relevant_properties=["specific_capacity", "coulombic_efficiency", "cycle_life",
                           "rate_capability", "initial_coulombic_efficiency"],
        typical_quantities={
            "hard_carbon_capacity": (200, 400, "mAh/g"),
            "hard_carbon_ice": (50, 85, "%"),
            "alloying_anode_capacity": (400, 700, "mAh/g"),
            "alloying_volume_expansion": (200, 500, "%"),
            "first_cycle_irreversibility": (15, 50, "%"),
        },
        example_queries=[
            "Why can't sodium intercalate into graphite like lithium does?",
            "What is the sodium storage mechanism in hard carbon anodes?",
            "How can we improve the initial Coulombic efficiency of hard carbon?",
            "Compare hard carbon vs. alloying anodes for SIBs in terms of volume expansion",
            "What pre-sodiation strategies can compensate for sodium loss in the first cycle?",
        ],
        visualization_focus=["anode_materials_subgraph", "ice_analysis", "volume_expansion_comparison"]
    ),

    SIBCoreProblem.CATHODE_INSTABILITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.CATHODE_INSTABILITY,
        title="Cathode Structural Instability and Volume Change",
        scientific_description=(
            "The larger Na⁺ ion (~55% larger than Li⁺) causes significant mechanical stress "
            "during insertion/extraction in cathode structures. In layered transition metal oxides "
            "(e.g., NaₓMO₂), this leads to severe volume changes and phase transitions (P2→O2, "
            "O3→P3) during cycling. These structural transformations cause cracking, capacity "
            "fade, and ultimately failure of the cathode material."
        ),
        root_cause="Na⁺ induces ~55% larger lattice parameter changes than Li⁺ during "
                   "intercalation/deintercalation, causing irreversible phase transitions",
        key_concepts=[
            "layered_oxide_cathode", "polyanionic_cathode", "prussian_blue_analogue",
            "nasicon_cathode", "phase_transition", "p2_o2_transition", "o3_p3_transition",
            "structural_degradation", "volume_change", "lattice_stabilization",
            "elemental_doping", "concentration_gradient"
        ],
        key_relationships=[
            ("layered_oxide_cathode", "CAUSES", "phase_transition"),
            ("phase_transition", "CAUSES", "cycle_life"),
            ("elemental_doping", "STABILIZES", "layered_oxide_cathode"),
            ("polyanionic_cathode", "INFLUENCES", "cycle_life"),
            ("prussian_blue_analogue", "INFLUENCES", "rate_capability"),
            ("volume_expansion", "CAUSES", "structural_degradation"),
        ],
        solution_directions=[
            "Elemental doping (Li, Mg, Zn, Ti) to stabilize layered structures",
            "Create concentration-gradient cathodes with stable surface",
            "Use polyanionic frameworks with robust 3D structures",
            "Design single-crystal cathodes to eliminate grain boundary cracking",
            "Develop Prussian blue analogues with minimal volume change"
        ],
        relevant_materials=["layered_oxide_cathode", "polyanionic_cathode",
                          "prussian_blue_analogue", "nasicon_cathode", "organic_cathode"],
        relevant_phenomena=["phase_transition", "volume_expansion", "intercalation",
                          "structural_degradation", "solid_solution_behavior"],
        relevant_properties=["specific_capacity", "cycle_life", "rate_capability",
                           "voltage_plateau", "energy_density"],
        typical_quantities={
            "layered_oxide_capacity": (100, 200, "mAh/g"),
            "polyanionic_capacity": (100, 120, "mAh/g"),
            "pba_capacity": (100, 170, "mAh/g"),
            "volume_change_layered": (5, 25, "%"),
            "phase_transition_voltage": (2.0, 4.2, "V vs. Na/Na⁺"),
        },
        example_queries=[
            "What causes the P2 to O2 phase transition in NaₓMnO₂ cathodes?",
            "How does elemental doping stabilize layered oxide cathodes for SIBs?",
            "Compare the structural stability of layered vs. polyanionic cathodes",
            "Why do single-crystal cathodes show better cycling stability?",
        ],
        visualization_focus=["cathode_phase_diagram", "doping_effects", "structural_comparison"]
    ),

    SIBCoreProblem.SEI_CHEMISTRY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SEI_CHEMISTRY,
        title="Electrolyte and Interphase (SEI) Chemistry",
        scientific_description=(
            "The Solid Electrolyte Interphase (SEI) is a protective layer formed on the anode "
            "from electrolyte decomposition. In SIBs, sodium salts (e.g., NaPF₆) are more soluble "
            "in the SEI layer compared to lithium salts, leading to continuous electrolyte "
            "consumption, high interfacial resistance, and poor cycle life."
        ),
        root_cause="Sodium salts have higher solubility in the SEI matrix than lithium salts, "
                   "preventing formation of a stable, self-limiting passivation layer",
        key_concepts=[
            "sei_formation", "liquid_electrolyte", "interface_engineering",
            "artificial_sei", "electrolyte_decomposition", "sei_solubility",
            "cei_formation", "interface_resistance", "electrolyte_formulation",
            "concentrated_electrolyte", "fluorinated_electrolyte"
        ],
        key_relationships=[
            ("liquid_electrolyte", "CAUSES", "sei_formation"),
            ("sei_formation", "INFLUENCES", "cycle_life"),
            ("sei_formation", "INFLUENCES", "coulombic_efficiency"),
            ("interface_engineering", "STABILIZES", "sei_formation"),
            ("interface_resistance", "INFLUENCES", "rate_capability"),
            ("electrolyte_decomposition", "CAUSES", "low_ice"),
        ],
        solution_directions=[
            "Design concentrated electrolytes to reduce free solvent and stabilize SEI",
            "Use fluorinated solvents/additives for robust SEI formation",
            "Create artificial SEI layers (Al₂O₃, polymer coatings)",
            "Develop new sodium salts with lower SEI solubility",
            "Optimize electrolyte formulation (solvent ratio, salt concentration)",
        ],
        relevant_materials=["liquid_electrolyte", "solid_electrolyte", "polymer_electrolyte",
                          "quasi_solid_electrolyte", "aqueous_electrolyte"],
        relevant_phenomena=["sei_formation", "electrolyte_decomposition", "sodium_plating_stripping"],
        relevant_properties=["coulombic_efficiency", "cycle_life", "ionic_conductivity"],
        typical_quantities={
            "sei_resistance": (50, 500, "Ω·cm²"),
            "na_sei_thickness": (10, 100, "nm"),
            "first_cycle_loss": (15, 40, "%"),
            "ionic_conductivity_liquid": (1e-3, 1e-2, "S/cm"),
        },
        example_queries=[
            "Why is the SEI in SIBs less stable than in LIBs?",
            "How do concentrated electrolytes improve SEI formation in sodium batteries?",
            "What role do fluorinated additives play in SEI stabilization?",
            "Compare artificial vs. natural SEI formation strategies",
        ],
        visualization_focus=["sei_composition_map", "resistance_evolution", "electrolyte_comparison"]
    ),

    SIBCoreProblem.SOLID_STATE_INTERFACE: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SOLID_STATE_INTERFACE,
        title="Solid-State and Semi-Solid Interface Challenges",
        scientific_description=(
            "Moving to solid-state or quasi-solid-state SIBs introduces severe interfacial problems. "
            "Solid electrolytes suffer from poor physical contact with electrodes, leading to high "
            "interfacial resistance. Sodium metal anodes grow dendritic structures that penetrate "
            "solid electrolytes, causing short circuits."
        ),
        root_cause="Rigid solid electrolytes cannot accommodate the volume changes of electrode "
                   "materials, causing contact loss, void formation, and dendrite penetration",
        key_concepts=[
            "solid_electrolyte", "quasi_solid_electrolyte", "solid_polymer_electrolyte",
            "interface_contact", "interfacial_resistance", "dendrite_growth",
            "void_formation", "delamination", "chemo_mechanical_mismatch",
            "all_solid_state_sodium_battery", "nasicon", "sulfide_electrolyte"
        ],
        key_relationships=[
            ("solid_electrolyte", "CAUSES", "interfacial_resistance"),
            ("dendrite_growth", "CAUSES", "short_circuit"),
            ("chemo_mechanical_mismatch", "CAUSES", "void_formation"),
            ("void_formation", "CAUSES", "delamination"),
            ("interface_engineering", "STABILIZES", "interface_contact"),
            ("quasi_solid_electrolyte", "INFLUENCES", "interface_contact"),
        ],
        solution_directions=[
            "Apply interfacial coating layers (oxide, polymer) to improve wetting",
            "Design compliant interlayers that accommodate volume change",
            "Use quasi-solid/gel electrolytes for better contact",
            "Develop dendrite-suppressing solid electrolytes",
            "Apply external pressure to maintain contact during cycling",
        ],
        relevant_materials=["solid_electrolyte", "quasi_solid_electrolyte",
                          "solid_polymer_electrolyte", "all_solid_state_sodium_battery"],
        relevant_phenomena=["dendrite_growth", "void_formation", "delamination",
                          "interfacial_resistance", "sodium_plating_stripping"],
        relevant_properties=["ionic_conductivity", "interfacial_resistance", "cycle_life"],
        typical_quantities={
            "solid_electrolyte_conductivity": (1e-5, 1e-3, "S/cm"),
            "interfacial_resistance": (10, 1000, "Ω·cm²"),
            "critical_current_density": (0.1, 2.0, "mA/cm²"),
            "stack_pressure": (1, 100, "MPa"),
        },
        example_queries=[
            "What causes void formation at the solid electrolyte-anode interface?",
            "How can quasi-solid electrolytes improve solid-state SIB performance?",
            "Compare NASICON vs. sulfide solid electrolytes for sodium batteries",
            "What strategies suppress sodium dendrite growth in solid electrolytes?",
        ],
        visualization_focus=["interface_schematic", "dendrite_penetration", "pressure_effects"]
    ),

    SIBCoreProblem.LOW_ENERGY_DENSITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.LOW_ENERGY_DENSITY,
        title="Lower Energy Density Challenge",
        scientific_description=(
            "Sodium has a lower standard reduction potential (-2.71 V vs SHE) compared to lithium "
            "(-3.04 V vs SHE), and sodium is heavier (23 vs. 7 g/mol). Consequently, SIBs "
            "inherently have a lower theoretical energy density than LIBs."
        ),
        root_cause="Na/Na⁺ potential is 0.33 V higher than Li/Li⁺, and Na atomic mass is 3.3× "
                   "that of Li, both reducing theoretical specific energy",
        key_concepts=[
            "energy_density", "specific_capacity", "voltage_plateau",
            "high_voltage_cathode", "high_capacity_anode", "full_cell",
            "inactive_components", "electrode_loading", "n_p_ratio",
            "cell_level_energy", "gravimetric_energy", "volumetric_energy"
        ],
        key_relationships=[
            ("specific_capacity", "CAUSES", "energy_density"),
            ("voltage_plateau", "INFLUENCES", "energy_density"),
            ("high_voltage_cathode", "INFLUENCES", "energy_density"),
            ("high_capacity_anode", "INFLUENCES", "energy_density"),
            ("full_cell", "DEPENDS_ON", "energy_density"),
            ("inactive_components", "CONSTRAINS", "energy_density"),
        ],
        solution_directions=[
            "Develop high-voltage cathodes (>4.0 V vs. Na/Na⁺)",
            "Use high-capacity anodes (alloying, conversion) with volume mitigation",
            "Optimize full-cell design (N/P ratio, electrode loading)",
            "Minimize inactive components (thin current collectors, lean electrolyte)",
            "Explore sodium metal anodes for maximum energy",
        ],
        relevant_materials=["layered_oxide_cathode", "polyanionic_cathode", "hard_carbon",
                          "alloying_anode", "sodium_metal", "full_cell", "organic_cathode"],
        relevant_phenomena=["intercalation", "conversion_reaction", "sodium_plating_stripping"],
        relevant_properties=["energy_density", "specific_capacity", "voltage_plateau"],
        typical_quantities={
            "sib_cell_energy": (100, 180, "Wh/kg"),
            "lib_cell_energy": (200, 300, "Wh/kg"),
            "high_voltage_cathode": (3.8, 4.5, "V vs. Na/Na⁺"),
            "target_ev_energy": (150, 200, "Wh/kg"),
        },
        example_queries=[
            "What is the theoretical energy density limit for SIBs vs LIBs?",
            "How can high-voltage cathodes push SIB energy density closer to LIBs?",
            "Compare full-cell energy densities with different anode-cathode pairs",
            "Can SIBs ever achieve energy density competitive with LIBs for EVs?",
        ],
        visualization_focus=["energy_density_comparison", "ragone_plot", "voltage_profile_comparison"]
    ),

    SIBCoreProblem.MOISTURE_MANUFACTURING: SIBProblemDefinition(
        problem_id=SIBCoreProblem.MOISTURE_MANUFACTURING,
        title="Moisture Sensitivity and Manufacturing Challenges",
        scientific_description=(
            "Many high-performance sodium cathode materials and sodium salts (NaPF₆) are highly "
            "hygroscopic—they absorb moisture from air. When exposed to humidity, these materials "
            "degrade, forming residual alkaline compounds (NaOH, Na₂CO₃) on the surface that cause "
            "electrode slurry to gel up and ruin the coating process."
        ),
        root_cause="Sodium cathode materials react with atmospheric H₂O/CO₂ to form surface "
                   "alkaline species (NaOH, Na₂CO₃) that disrupt slurry rheology",
        key_concepts=[
            "moisture_sensitivity", "hygroscopic_materials", "surface_alkalinity",
            "slurry_gelation", "coating_defects", "aqueous_processing",
            "moisture_stable_materials", "surface_washing", "dry_room_requirements",
            "manufacturing_scalability", "electrode_fabrication", "slurry_coating"
        ],
        key_relationships=[
            ("moisture_sensitivity", "CAUSES", "surface_alkalinity"),
            ("surface_alkalinity", "CAUSES", "slurry_gelation"),
            ("slurry_gelation", "CAUSES", "coating_defects"),
            ("surface_washing", "REDUCES", "surface_alkalinity"),
            ("aqueous_processing", "ENABLES", "manufacturing_scalability"),
            ("slurry_coating", "PROCESSES", "electrode_fabrication"),
        ],
        solution_directions=[
            "Develop moisture-stable cathode compositions (surface doping, coating)",
            "Apply surface washing treatments to remove alkaline compounds",
            "Use aqueous binders (CMC, alginate) compatible with alkaline surfaces",
            "Design dry room specifications optimized for sodium materials",
            "Explore water-based slurry processing with pH control",
        ],
        relevant_materials=["layered_oxide_cathode", "prussian_blue_analogue", "polyanionic_cathode"],
        relevant_phenomena=["surface_alkalinity", "slurry_gelation", "moisture_degradation"],
        relevant_properties=["coulombic_efficiency", "cycle_life", "manufacturing_yield"],
        typical_quantities={
            "surface_pH": (10, 13, "pH units"),
            "na2co3_content": (0.5, 5.0, "wt%"),
            "dry_room_dew_point": (-40, -60, "°C"),
            "slurry_viscosity_target": (1000, 5000, "mPa·s"),
        },
        example_queries=[
            "Why are sodium cathode materials more moisture-sensitive than lithium cathodes?",
            "How does surface alkalinity affect slurry coating quality?",
            "What surface washing treatments can stabilize sodium cathode materials?",
            "Compare dry room requirements for SIB vs LIB manufacturing",
        ],
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
# COMPLETE DOMAIN ONTOLOGY
# ============================================================================
class DomainOntology:
    """Complete ontology for Sodium-Ion Battery concepts."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self.synonym_to_canonical: Dict[str, str] = {}
        self._build_ontology()

    def _add_concept(
        self,
        canonical_name: str,
        concept_type: ConceptType,
        synonyms: Set[str] = None,
        definition: str = "",
    ) -> None:
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            definition=definition,
        )
        self.concepts[canonical_name] = node

    def _add_relationship(
        self,
        source: str,
        rel_type: RelationshipType,
        target: str,
        confidence: float = 1.0,
    ) -> None:
        self.relationships.append(Relationship(source, target, rel_type, confidence))

    def _build_ontology(self) -> None:
        # === ANODE MATERIALS ===
        self._add_concept("hard_carbon", ConceptType.MATERIAL,
            synonyms={"hc", "hard carbon anode", "disordered carbon", "non-graphitizable carbon"},
            definition="Hard carbon with disordered structure, the most common sodium-ion battery anode")
        self._add_concept("alloying_anode", ConceptType.MATERIAL,
            synonyms={"sn anode", "sb anode", "bi anode", "tin anode", "antimony anode", "alloy anode"},
            definition="Alloying-type anode materials (Sn, Sb, Bi) with high capacity but large volume change")
        self._add_concept("intercalation_anode", ConceptType.MATERIAL,
            synonyms={"tio2", "na2ti3o7", "layered titanium oxide"},
            definition="Intercalation anode materials (e.g., TiO₂, Na₂Ti₃O₇) with stable cycling")
        self._add_concept("sodium_metal", ConceptType.MATERIAL,
            synonyms={"na metal", "sodium anode", "metallic sodium", "na foil"},
            definition="Pure sodium metal anode for high energy density")
        self._add_concept("mxene", ConceptType.MATERIAL,
            synonyms={"mxenes", "ti3c2tx", "v2ctz", "2d transition metal carbide"},
            definition="MXenes, 2D transition metal carbides/nitrides for high-rate SIB anodes")
        self._add_concept("conversion_anode", ConceptType.MATERIAL,
            synonyms={"conversion reaction anode", "metal sulfide anode", "fes2", "cos2", "mos2"},
            definition="Conversion-type anode materials with high capacity")

        # === CATHODE MATERIALS ===
        self._add_concept("layered_oxide_cathode", ConceptType.MATERIAL,
            synonyms={"na_mno2", "namno2", "na_x_mno2", "p2_na_mno2", "o3_na_mno2", "layered oxide"},
            definition="Sodium transition metal oxide cathodes with layered structure")
        self._add_concept("polyanionic_cathode", ConceptType.MATERIAL,
            synonyms={"na3v2(po4)3", "nvp", "na3v2(po4)2f3", "nvpf", "polyanion"},
            definition="Polyanionic compound cathodes with NASICON or phosphate frameworks")
        self._add_concept("prussian_blue_analogue", ConceptType.MATERIAL,
            synonyms={"pba", "prussian blue", "na2mnfe(cn)6", "hexacyanoferrate"},
            definition="Prussian blue analogues with open framework for sodium intercalation")
        self._add_concept("nasicon_cathode", ConceptType.MATERIAL,
            synonyms={"nasicon", "na superionic conductor", "na3zr2si2po12"},
            definition="NASICON-type cathodes with 3D framework for fast sodium transport")
        self._add_concept("organic_cathode", ConceptType.MATERIAL,
            synonyms={"organic electrode", "pdtca", "ppta", "conjugated carbonyl"},
            definition="Organic cathode materials with structural flexibility")

        # === ELECTROLYTES ===
        self._add_concept("liquid_electrolyte", ConceptType.MATERIAL,
            synonyms={"organic electrolyte", "naclo4 in ec/dec", "na pf6", "aqueous electrolyte"},
            definition="Liquid electrolyte for sodium-ion batteries")
        self._add_concept("solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"solid sodium electrolyte", "nasicon", "na3ps4", "sulfide electrolyte"},
            definition="Solid-state electrolyte for all-solid-state sodium batteries")
        self._add_concept("polymer_electrolyte", ConceptType.MATERIAL,
            synonyms={"peo", "polyethylene oxide", "gel polymer"},
            definition="Polymer-based electrolyte with sodium salt")
        self._add_concept("quasi_solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"gel electrolyte", "quasi-solid", "semi-solid", "in-situ polymerized"},
            definition="Quasi-solid electrolyte blending polymer and liquid")
        self._add_concept("aqueous_electrolyte", ConceptType.MATERIAL,
            synonyms={"aqueous na electrolyte", "water based electrolyte", "na2so4 electrolyte"},
            definition="Aqueous electrolyte using water as solvent")
        self._add_concept("solid_polymer_electrolyte", ConceptType.MATERIAL,
            synonyms={"spe", "pan based electrolyte", "pmma electrolyte"},
            definition="Solid polymer electrolyte for flexible all-solid-state SIBs")

        # === PROPERTIES ===
        self._add_concept("specific_capacity", ConceptType.PROPERTY,
            synonyms={"capacity", "mah/g", "specific charge", "gravimetric capacity"},
            definition="Specific capacity (mAh/g) of electrode material")
        self._add_concept("energy_density", ConceptType.PROPERTY,
            synonyms={"wh/kg", "specific energy", "volumetric energy density"},
            definition="Energy density (Wh/kg) of the full cell or electrode")
        self._add_concept("coulombic_efficiency", ConceptType.PROPERTY,
            synonyms={"ce", "charge-discharge efficiency", "reversibility"},
            definition="Coulombic efficiency (%), ratio of discharge to charge capacity")
        self._add_concept("cycle_life", ConceptType.PROPERTY,
            synonyms={"cycling stability", "retention", "capacity retention"},
            definition="Cycle life before capacity drops below 80%")
        self._add_concept("rate_capability", ConceptType.PROPERTY,
            synonyms={"rate performance", "high rate", "c-rate"},
            definition="Ability to maintain capacity at high charge/discharge rates")
        self._add_concept("ionic_conductivity", ConceptType.PROPERTY,
            synonyms={"na+ conductivity", "s/cm", "ionic transport"},
            definition="Ionic conductivity (S/cm) of electrolyte or electrode")
        self._add_concept("voltage_plateau", ConceptType.PROPERTY,
            synonyms={"discharge voltage", "charge voltage", "voltage profile"},
            definition="Voltage plateau (V) during discharge/charge")
        self._add_concept("interface_resistance", ConceptType.PROPERTY,
            synonyms={"interfacial resistance", "contact resistance"},
            definition="Resistance at electrode-electrolyte interface")

        # === PHENOMENA ===
        self._add_concept("dendrite_growth", ConceptType.PHENOMENON,
            synonyms={"sodium dendrite", "dendrite formation", "mossy sodium"},
            definition="Formation of sodium dendrites during plating")
        self._add_concept("sei_formation", ConceptType.PHENOMENON,
            synonyms={"solid electrolyte interphase", "sei layer", "passivation film"},
            definition="Solid-electrolyte interphase formed on anode")
        self._add_concept("sodium_plating_stripping", ConceptType.PHENOMENON,
            synonyms={"na plating", "sodium stripping", "electrodeposition"},
            definition="Electrochemical deposition and dissolution of sodium metal")
        self._add_concept("intercalation", ConceptType.PHENOMENON,
            synonyms={"na+ insertion", "sodium intercalation", "deintercalation"},
            definition="Insertion/extraction of Na+ ions into host electrode structure")
        self._add_concept("conversion_reaction", ConceptType.PHENOMENON,
            synonyms={"conversion", "alloying/dealloying", "conversion electrode"},
            definition="Electrochemical conversion reaction")
        self._add_concept("volume_expansion", ConceptType.PHENOMENON,
            synonyms={"structural change", "lattice expansion", "pulverization"},
            definition="Volume expansion and mechanical degradation during cycling")
        self._add_concept("phase_transition", ConceptType.PHENOMENON,
            synonyms={"p2_o2_transition", "o3_p3_transition", "structural transition"},
            definition="Phase transitions in cathode materials during cycling")
        self._add_concept("thermal_runaway", ConceptType.PHENOMENON,
            synonyms={"thermal abuse", "overheating", "battery fire"},
            definition="Thermal runaway and safety-related thermal phenomena")
        self._add_concept("void_formation", ConceptType.PHENOMENON,
            synonyms={"contact loss", "delamination", "gap formation"},
            definition="Void formation at solid-state interfaces")

        # === METHODS ===
        self._add_concept("cyclic_voltammetry", ConceptType.METHOD,
            synonyms={"cv", "cyclic voltammogram", "voltammetry"},
            definition="Cyclic voltammetry for electrochemical characterization")
        self._add_concept("electrochemical_impedance_spectroscopy", ConceptType.METHOD,
            synonyms={"eis", "nyquist plot", "impedance spectroscopy"},
            definition="EIS for interface and kinetics characterization")
        self._add_concept("galvanostatic_cycling", ConceptType.METHOD,
            synonyms={"constant current", "cccv", "galvanostatic"},
            definition="Galvanostatic cycling at constant current")
        self._add_concept("operando_characterization", ConceptType.METHOD,
            synonyms={"in situ xrd", "operando xrd", "in situ raman"},
            definition="Real-time characterization during battery operation")

        # === PROCESSES ===
        self._add_concept("slurry_coating", ConceptType.PROCESS,
            synonyms={"electrode coating", "doctor blade", "tape casting"},
            definition="Slurry coating process for electrode fabrication")
        self._add_concept("interface_engineering", ConceptType.PROCESS,
            synonyms={"surface coating", "interfacial layer", "artificial sei"},
            definition="Surface and interface engineering strategies")
        self._add_concept("pre_sodiation", ConceptType.PROCESS,
            synonyms={"pre-sodiation", "sodium compensation", "sacrificial salt"},
            definition="Pre-sodiation to compensate for initial sodium loss")

        # === DEVICES ===
        self._add_concept("sodium_ion_battery", ConceptType.MODEL,
            synonyms={"sib", "na-ion battery", "sodium battery"},
            definition="Sodium-ion battery system")
        self._add_concept("all_solid_state_sodium_battery", ConceptType.MODEL,
            synonyms={"asssb", "solid-state sodium"},
            definition="All-solid-state sodium battery")
        self._add_concept("full_cell", ConceptType.MODEL,
            synonyms={"full sodium ion cell", "practical cell"},
            definition="Practical full-cell configuration")

        # Build synonym index
        self._build_synonym_index()
        
        # Build relationships
        self._build_relationships()

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_relationships(self) -> None:
        # Anode relationships
        self._add_relationship("hard_carbon", RelationshipType.INFLUENCES, "specific_capacity", 0.85)
        self._add_relationship("hard_carbon", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.75)
        self._add_relationship("hard_carbon", RelationshipType.INFLUENCES, "cycle_life", 0.70)
        self._add_relationship("alloying_anode", RelationshipType.INFLUENCES, "specific_capacity", 0.90)
        self._add_relationship("alloying_anode", RelationshipType.CAUSES, "volume_expansion", 0.85)
        self._add_relationship("volume_expansion", RelationshipType.CAUSES, "cycle_life", -0.80)
        self._add_relationship("sodium_metal", RelationshipType.INFLUENCES, "energy_density", 0.90)
        self._add_relationship("sodium_metal", RelationshipType.CAUSES, "dendrite_growth", 0.80)
        self._add_relationship("mxene", RelationshipType.INFLUENCES, "rate_capability", 0.90)
        self._add_relationship("pre_sodiation", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.85)

        # Cathode relationships
        self._add_relationship("layered_oxide_cathode", RelationshipType.INFLUENCES, "specific_capacity", 0.80)
        self._add_relationship("layered_oxide_cathode", RelationshipType.CAUSES, "phase_transition", 0.75)
        self._add_relationship("phase_transition", RelationshipType.CAUSES, "cycle_life", -0.70)
        self._add_relationship("polyanionic_cathode", RelationshipType.INFLUENCES, "cycle_life", 0.85)
        self._add_relationship("polyanionic_cathode", RelationshipType.INFLUENCES, "voltage_plateau", 0.80)
        self._add_relationship("prussian_blue_analogue", RelationshipType.INFLUENCES, "rate_capability", 0.80)
        self._add_relationship("organic_cathode", RelationshipType.INFLUENCES, "specific_capacity", 0.75)

        # Electrolyte relationships
        self._add_relationship("liquid_electrolyte", RelationshipType.CAUSES, "sei_formation", 0.85)
        self._add_relationship("solid_electrolyte", RelationshipType.INFLUENCES, "ionic_conductivity", 0.90)
        self._add_relationship("solid_electrolyte", RelationshipType.INFLUENCES, "dendrite_growth", -0.60)
        self._add_relationship("solid_electrolyte", RelationshipType.CAUSES, "interface_resistance", 0.70)
        self._add_relationship("quasi_solid_electrolyte", RelationshipType.INFLUENCES, "interface_resistance", -0.50)
        self._add_relationship("aqueous_electrolyte", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.70)

        # SEI relationships
        self._add_relationship("sei_formation", RelationshipType.INFLUENCES, "cycle_life", 0.70)
        self._add_relationship("sei_formation", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.75)
        self._add_relationship("interface_engineering", RelationshipType.STABILIZES, "sei_formation", 0.80)

        # Property relationships
        self._add_relationship("specific_capacity", RelationshipType.CAUSES, "energy_density", 0.95)
        self._add_relationship("voltage_plateau", RelationshipType.INFLUENCES, "energy_density", 0.90)
        self._add_relationship("ionic_conductivity", RelationshipType.INFLUENCES, "rate_capability", 0.85)
        self._add_relationship("coulombic_efficiency", RelationshipType.CAUSES, "cycle_life", 0.85)

        # Method relationships
        self._add_relationship("cyclic_voltammetry", RelationshipType.DETECTS, "intercalation", 0.85)
        self._add_relationship("electrochemical_impedance_spectroscopy", RelationshipType.DETECTS, "sei_formation", 0.80)
        self._add_relationship("galvanostatic_cycling", RelationshipType.MEASURES, "specific_capacity", 0.90)
        self._add_relationship("operando_characterization", RelationshipType.OBSERVES, "phase_transition", 0.85)

        # Process relationships
        self._add_relationship("slurry_coating", RelationshipType.PROCESSES, "full_cell", 0.85)
        self._add_relationship("interface_engineering", RelationshipType.IMPROVES, "cycle_life", 0.75)

        # Device relationships
        self._add_relationship("all_solid_state_sodium_battery", RelationshipType.HYPONYM, "sodium_ion_battery", 0.9)
        self._add_relationship("full_cell", RelationshipType.HYPONYM, "sodium_ion_battery", 0.95)
        self._add_relationship("full_cell", RelationshipType.DEPENDS_ON, "coulombic_efficiency", 0.80)

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        return None

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].concept_type
        return ConceptType.GENERAL

    def get_definition(self, canonical_name: str) -> str:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def get_related_concepts(
        self, canonical_name: str, rel_type: RelationshipType = None
    ) -> List[Tuple[str, RelationshipType, float]]:
        related: List[Tuple[str, RelationshipType, float]] = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related


# ============================================================================
# SPARSE GRAPHSAGE GNN (Complete Implementation)
# ============================================================================
class SparseGraphSAGE(nn.Module):
    """Sparse GraphSAGE for concept graph node embeddings."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregator: str = "mean",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = hidden_dim if i > 0 else hidden_dim
            out_features = out_dim if i == num_layers - 1 else hidden_dim
            self.sage_layers.append(SAGEConv(in_features, out_features, aggregator))

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else out_dim)
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)

        for i, (sage, bn) in enumerate(zip(self.sage_layers, self.batch_norms)):
            h_new = sage(h, edge_index, edge_weight)
            h_new = bn(h_new)
            if i < self.num_layers - 1:
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new

        return h


class SAGEConv(nn.Module):
    """GraphSAGE convolution layer with sparse operations."""

    def __init__(self, in_dim: int, out_dim: int, aggregator: str = "mean"):
        super().__init__()
        self.aggregator = aggregator
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self transformation
        self_out = self.self_linear(x)

        # Neighbor aggregation using sparse operations
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        # Normalize edge weights
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, edge_weight)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        norm_weight = edge_weight * deg_inv[row]

        # Sparse matrix multiplication for neighbor aggregation
        indices = torch.stack([row, col])
        values = norm_weight
        size = (num_nodes, num_nodes)
        adj_sparse = torch.sparse_coo_tensor(indices, values, size)

        if self.aggregator == "mean":
            neigh_out = torch.sparse.mm(adj_sparse, x)
        elif self.aggregator == "sum":
            neigh_out = torch.sparse.mm(adj_sparse.t(), x * edge_weight.unsqueeze(1))
        else:
            neigh_out = torch.sparse.mm(adj_sparse, x)

        neigh_out = self.neigh_linear(neigh_out)

        return self_out + neigh_out


# ============================================================================
# LLM QUERY ANALYZER (Complete Implementation)
# ============================================================================
SIB_SYSTEM_PROMPT = """You are an expert Sodium-Ion Battery (SIB) research assistant.
Analyze queries and classify them according to the six core SIB challenges:
1. ANODE_BOTTLENECK: Graphite incompatibility, hard carbon ICE, alloying volume expansion
2. CATHODE_INSTABILITY: Phase transitions (P2→O2), structural degradation
3. SEI_CHEMISTRY: Unstable SEI, electrolyte decomposition
4. SOLID_STATE_INTERFACE: Poor contact, dendrite penetration, void formation
5. LOW_ENERGY_DENSITY: Lower voltage, heavier Na
6. MOISTURE_MANUFACTURING: Hygroscopic materials, slurry gelation
Output valid JSON only."""


class SIBQueryAnalyzer:
    """LLM-directed query analyzer for SIB Concept Graph."""

    def __init__(
        self,
        ontology: DomainOntology,
        mode: str = "rule_based",
        model_name: str = "gpt2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_model: str = "gpt-3.5-turbo",
    ):
        self.ontology = ontology
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_model = api_model
        self._llm = None
        self._tokenizer = None
        self._initialized = False

        if mode == "local" and TRANSFORMERS_AVAILABLE:
            self._init_local_llm()
        elif mode == "api" and OPENAI_AVAILABLE and api_key:
            self._init_api_llm()
        else:
            self.mode = "rule_based"
            self._initialized = True

    def _init_local_llm(self):
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._llm.eval()
            self._initialized = True
        except Exception as e:
            print(f"Failed to init local LLM: {e}")
            self.mode = "rule_based"
            self._initialized = True

    def _init_api_llm(self):
        try:
            if self.api_base:
                openai.api_base = self.api_base
            openai.api_key = self.api_key
            self._initialized = True
        except Exception as e:
            print(f"Failed to init API LLM: {e}")
            self.mode = "rule_based"
            self._initialized = True

    def analyze_query(self, query: str) -> QueryAnalysisResult:
        if self.mode == "api" and self._initialized:
            return self._analyze_with_api(query)
        elif self.mode == "local" and self._initialized:
            return self._analyze_with_local_llm(query)
        else:
            return self._analyze_rule_based(query)

    def _analyze_with_api(self, query: str) -> QueryAnalysisResult:
        prompt = self._build_analysis_prompt(query)
        try:
            response = openai.ChatCompletion.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": SIB_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            result_text = response.choices[0].message.content
            return self._parse_llm_response(query, result_text)
        except Exception as e:
            print(f"API error: {e}")
            return self._analyze_rule_based(query)

    def _analyze_with_local_llm(self, query: str) -> QueryAnalysisResult:
        prompt = self._build_analysis_prompt(query)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            with torch.no_grad():
                outputs = self._llm.generate(**inputs, max_new_tokens=500, temperature=0.1, do_sample=False)
            result_text = self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return self._parse_llm_response(query, result_text)
        except Exception as e:
            print(f"Local LLM error: {e}")
            return self._analyze_rule_based(query)

    def _analyze_rule_based(self, query: str) -> QueryAnalysisResult:
        query_lower = query.lower()

        problem_scores = self._score_problems_rule_based(query_lower)
        primary_problem = max(problem_scores, key=problem_scores.get)
        secondary_problems = [
            p for p, score in sorted(problem_scores.items(), key=lambda x: x[1], reverse=True)
            if p != primary_problem and score > 0.3
        ][:2]

        if len(secondary_problems) >= 2 and all(
            problem_scores[p] > 0.5 for p in [primary_problem] + secondary_problems
        ):
            primary_problem = SIBCoreProblem.MULTI_PROBLEM

        matched_concepts = self._match_concepts(query_lower)
        inferred_concepts = self._infer_concepts(query_lower, primary_problem, matched_concepts)
        emphasis_direction = self._detect_emphasis_direction(query_lower)
        emphasis_keywords = self._extract_emphasis_keywords(query_lower)
        subgraph_depth, include_phenomena, include_solutions, include_quantities = \
            self._determine_scope(query_lower, emphasis_direction)

        if primary_problem != SIBCoreProblem.MULTI_PROBLEM:
            suggested_vis = SIB_PROBLEM_DEFINITIONS[primary_problem].visualization_focus
        else:
            suggested_vis = ["multi_problem_comparison"]

        reasoning = self._build_reasoning(query, primary_problem, matched_concepts, emphasis_direction)
        all_relevant = list(set(matched_concepts + inferred_concepts))

        return QueryAnalysisResult(
            original_query=query,
            primary_problem=primary_problem,
            secondary_problems=secondary_problems,
            problem_confidences=problem_scores,
            matched_concepts=matched_concepts,
            inferred_concepts=inferred_concepts,
            all_relevant_concepts=all_relevant,
            emphasis_keywords=emphasis_keywords,
            emphasis_direction=emphasis_direction,
            subgraph_depth=subgraph_depth,
            include_phenomena=include_phenomena,
            include_solutions=include_solutions,
            include_quantities=include_quantities,
            reasoning=reasoning,
            suggested_visualizations=suggested_vis,
            graph_construction_params={
                "focus_concepts": all_relevant,
                "depth": subgraph_depth,
                "include_phenomena": include_phenomena,
                "include_solutions": include_solutions,
            }
        )

    def _score_problems_rule_based(self, query_lower: str) -> Dict[SIBCoreProblem, float]:
        scores = {p: 0.0 for p in SIBCoreProblem}
        problem_keywords = {
            SIBCoreProblem.ANODE_BOTTLENECK: [
                ("anode", 0.4), ("hard carbon", 0.5), ("graphite", 0.6),
                ("ice", 0.5), ("coulombic efficiency", 0.5), ("first cycle", 0.4),
                ("pore", 0.3), ("alloying", 0.5), ("tin", 0.3), ("antimony", 0.4),
                ("volume expansion", 0.4), ("storage mechanism", 0.5), ("pre-sodiation", 0.5),
            ],
            SIBCoreProblem.CATHODE_INSTABILITY: [
                ("cathode", 0.4), ("layered oxide", 0.6), ("phase transition", 0.7),
                ("p2", 0.5), ("o2", 0.5), ("structural", 0.4), ("doping", 0.5),
                ("degradation", 0.4), ("cracking", 0.4), ("polyanionic", 0.6),
                ("prussian blue", 0.5), ("concentration gradient", 0.5), ("single-crystal", 0.5),
            ],
            SIBCoreProblem.SEI_CHEMISTRY: [
                ("sei", 0.7), ("solid electrolyte interphase", 0.8), ("interface", 0.3),
                ("electrolyte", 0.4), ("decomposition", 0.5), ("solubility", 0.5),
                ("concentrated electrolyte", 0.6), ("fluorinated", 0.4), ("artificial sei", 0.6),
                ("cei", 0.5), ("passivation", 0.4),
            ],
            SIBCoreProblem.SOLID_STATE_INTERFACE: [
                ("solid-state", 0.6), ("solid electrolyte", 0.7), ("all-solid", 0.7),
                ("nasicon", 0.6), ("sulfide", 0.4), ("dendrite", 0.5),
                ("void", 0.5), ("delamination", 0.6), ("pressure", 0.4),
                ("quasi-solid", 0.5), ("chemo-mechanical", 0.7),
            ],
            SIBCoreProblem.LOW_ENERGY_DENSITY: [
                ("energy density", 0.7), ("wh/kg", 0.6), ("specific energy", 0.6),
                ("high-voltage", 0.5), ("ev", 0.3), ("electric vehicle", 0.4),
                ("gravimetric", 0.4), ("volumetric", 0.4), ("full cell", 0.4),
                ("n/p ratio", 0.5), ("electrode loading", 0.4),
            ],
            SIBCoreProblem.MOISTURE_MANUFACTURING: [
                ("moisture", 0.7), ("humidity", 0.6), ("hygroscopic", 0.7),
                ("slurry", 0.5), ("coating", 0.4), ("manufacturing", 0.5),
                ("alkalinity", 0.6), ("naoh", 0.6), ("na2co3", 0.6),
                ("dry room", 0.6), ("surface washing", 0.6), ("binder", 0.4),
            ],
        }

        for problem, keywords in problem_keywords.items():
            for keyword_data in keywords:
                if isinstance(keyword_data, tuple):
                    keyword, weight = keyword_data
                else:
                    keyword, weight = keyword_data, 0.5
                if keyword in query_lower:
                    scores[problem] += weight

        max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0
        scores = {p: min(score / max_score, 1.0) for p, score in scores.items()}

        if max(scores[p] for p in list(SIBCoreProblem)[:6]) > 0.3:
            scores[SIBCoreProblem.GENERAL] = 0.1

        return scores

    def _match_concepts(self, query_lower: str) -> List[str]:
        matched = []
        for concept_name, concept_node in self.ontology.concepts.items():
            if concept_node.is_match(query_lower):
                matched.append(concept_name)
            else:
                for syn in concept_node.synonyms:
                    if syn in query_lower:
                        matched.append(concept_name)
                        break
        return matched

    def _infer_concepts(
        self, query_lower: str, primary_problem: SIBCoreProblem, matched_concepts: List[str]
    ) -> List[str]:
        inferred = []
        if primary_problem in SIB_PROBLEM_DEFINITIONS:
            problem_def = SIB_PROBLEM_DEFINITIONS[primary_problem]
            for concept in problem_def.key_concepts:
                if concept not in matched_concepts and concept in self.ontology.concepts:
                    inferred.append(concept)
        for concept in matched_concepts:
            related = self.ontology.get_related_concepts(concept)
            for rel_concept, _, confidence in related:
                if confidence > 0.6 and rel_concept not in matched_concepts + inferred:
                    if rel_concept in self.ontology.concepts:
                        inferred.append(rel_concept)
        return inferred[:10]

    def _detect_emphasis_direction(self, query_lower: str) -> str:
        directions = {
            "causes": ["why", "cause", "reason", "lead to", "due to", "because"],
            "solutions": ["how to", "solve", "improve", "mitigate", "prevent", "strategy"],
            "comparison": ["compare", "vs", "versus", "difference", "between"],
            "mechanism": ["mechanism", "how does", "process", "pathway"],
            "quantitative": ["how much", "value", "number", "percentage", "measure"],
        }
        scores = {d: 0 for d in directions}
        for direction, keywords in directions.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[direction] += 1
        if max(scores.values()) == 0:
            return "general"
        return max(scores, key=scores.get)

    def _extract_emphasis_keywords(self, query_lower: str) -> List[str]:
        patterns = [r"especially\s+(\w+)", r"particularly\s+(\w+)", r"specifically\s+(\w+)",
                   r"mainly\s+(\w+)", r"focus\s+on\s+(\w+)", r"key\s+(\w+)"]
        keywords = []
        for pattern in patterns:
            keywords.extend(re.findall(pattern, query_lower))
        return keywords

    def _determine_scope(self, query_lower: str, emphasis_direction: str) -> Tuple[int, bool, bool, bool]:
        subgraph_depth = 2
        if "comprehensive" in query_lower or "detailed" in query_lower:
            subgraph_depth = 3
        elif "specific" in query_lower:
            subgraph_depth = 1
        include_phenomena = "mechanism" in emphasis_direction
        include_solutions = "solutions" in emphasis_direction
        include_quantities = "quantitative" in emphasis_direction
        return subgraph_depth, include_phenomena, include_solutions, include_quantities

    def _build_reasoning(self, query: str, primary_problem: SIBCoreProblem,
                        matched_concepts: List[str], emphasis_direction: str) -> str:
        if primary_problem == SIBCoreProblem.MULTI_PROBLEM:
            return f"Query spans multiple SIB problems. Concepts: {', '.join(matched_concepts[:5])}"
        title = SIB_PROBLEM_DEFINITIONS[primary_problem].title
        return f"Classified as: {title}. Matched {len(matched_concepts)} concepts. Focus: {emphasis_direction}."

    def _build_analysis_prompt(self, query: str) -> str:
        return f"""Analyze this SIB query and output JSON:
QUERY: "{query}"
OUTPUT: {{
    "primary_problem": "enum_value",
    "secondary_problems": ["enum_value"],
    "problem_confidences": {{"enum_value": 0.0}},
    "matched_concepts": ["concept"],
    "inferred_concepts": ["concept"],
    "emphasis_direction": "causes|solutions|comparison|mechanism|quantitative",
    "emphasis_keywords": ["kw"],
    "subgraph_depth": 2,
    "reasoning": "explanation"
}}"""

    def _parse_llm_response(self, query: str, response_text: str) -> QueryAnalysisResult:
        try:
            json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return self._analyze_rule_based(query)

            try:
                primary_problem = SIBCoreProblem[data["primary_problem"]]
            except (KeyError, ValueError):
                primary_problem = SIBCoreProblem.GENERAL

            secondary_problems = []
            for p_str in data.get("secondary_problems", []):
                try:
                    secondary_problems.append(SIBCoreProblem[p_str])
                except (KeyError, ValueError):
                    pass

            problem_confidences = {}
            for p_str, conf in data.get("problem_confidences", {}).items():
                try:
                    problem_confidences[SIBCoreProblem[p_str]] = float(conf)
                except (KeyError, ValueError):
                    pass

            return QueryAnalysisResult(
                original_query=query,
                primary_problem=primary_problem,
                secondary_problems=secondary_problems,
                problem_confidences=problem_confidences,
                matched_concepts=data.get("matched_concepts", []),
                inferred_concepts=data.get("inferred_concepts", []),
                all_relevant_concepts=data.get("matched_concepts", []) + data.get("inferred_concepts", []),
                emphasis_keywords=data.get("emphasis_keywords", []),
                emphasis_direction=data.get("emphasis_direction", "general"),
                subgraph_depth=data.get("subgraph_depth", 2),
                include_phenomena=data.get("include_phenomena", True),
                include_solutions=data.get("include_solutions", True),
                include_quantities=data.get("include_quantities", False),
                reasoning=data.get("reasoning", ""),
                suggested_visualizations=data.get("suggested_visualizations", []),
            )
        except Exception as e:
            print(f"Parse error: {e}")
            return self._analyze_rule_based(query)


# ============================================================================
# DIRECTED CONCEPT GRAPH BUILDER (Complete Implementation)
# ============================================================================
class DirectedConceptGraphBuilder:
    """Builds query-focused concept subgraphs."""

    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology

    def build_focused_subgraph(self, analysis: QueryAnalysisResult) -> nx.Graph:
        G = nx.DiGraph()
        focus_concepts = set(analysis.all_relevant_concepts)

        if analysis.primary_problem in SIB_PROBLEM_DEFINITIONS:
            problem_def = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem]
            focus_concepts.update(problem_def.get_ontology_concepts())

        for sec_problem in analysis.secondary_problems:
            if sec_problem in SIB_PROBLEM_DEFINITIONS:
                focus_concepts.update(SIB_PROBLEM_DEFINITIONS[sec_problem].get_ontology_concepts())

        valid_concepts = {c for c in focus_concepts if c in self.ontology.concepts}
        expanded_concepts = self._expand_concepts(valid_concepts, analysis.subgraph_depth)

        for concept_name in expanded_concepts:
            if concept_name in self.ontology.concepts:
                node = self.ontology.concepts[concept_name]
                relevance = self._calculate_relevance(concept_name, analysis)
                G.add_node(
                    concept_name,
                    concept_type=node.concept_type.value,
                    definition=node.definition,
                    relevance=relevance,
                    is_matched=concept_name in analysis.matched_concepts,
                    is_inferred=concept_name in analysis.inferred_concepts,
                )

        for rel in self.ontology.relationships:
            if rel.source in expanded_concepts and rel.target in expanded_concepts:
                edge_weight = self._calculate_edge_weight(rel, analysis)
                G.add_edge(
                    rel.source, rel.target,
                    rel_type=rel.rel_type.value,
                    confidence=rel.confidence,
                    weight=edge_weight,
                    color=get_edge_color(rel.rel_type),
                    width=get_edge_width(rel.rel_type) * edge_weight,
                    style=get_edge_style(rel.rel_type),
                )

        if analysis.include_solutions:
            self._add_solution_edges(G, analysis, expanded_concepts)

        return G

    def _expand_concepts(self, seed_concepts: Set[str], depth: int) -> Set[str]:
        expanded = set(seed_concepts)
        current_level = set(seed_concepts)
        for _ in range(depth):
            next_level = set()
            for concept in current_level:
                for rel_concept, _, confidence in self.ontology.get_related_concepts(concept):
                    if confidence > 0.5 and rel_concept in self.ontology.concepts:
                        next_level.add(rel_concept)
            new_concepts = next_level - expanded
            if not new_concepts:
                break
            expanded.update(new_concepts)
            current_level = new_concepts
        return expanded

    def _calculate_relevance(self, concept_name: str, analysis: QueryAnalysisResult) -> float:
        score = 0.0
        if concept_name in analysis.matched_concepts:
            score += 1.0
        elif concept_name in analysis.inferred_concepts:
            score += 0.7
        if analysis.primary_problem in SIB_PROBLEM_DEFINITIONS:
            problem_def = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem]
            if concept_name in problem_def.key_concepts:
                score += 0.8
            if concept_name in problem_def.relevant_materials:
                score += 0.6
        return min(score, 1.0)

    def _calculate_edge_weight(self, rel: Relationship, analysis: QueryAnalysisResult) -> float:
        base_weight = rel.confidence
        if rel.source in analysis.matched_concepts and rel.target in analysis.matched_concepts:
            base_weight *= 1.5
        elif rel.source in analysis.matched_concepts or rel.target in analysis.matched_concepts:
            base_weight *= 1.2
        return min(base_weight, 1.0)

    def _add_solution_edges(self, G: nx.Graph, analysis: QueryAnalysisResult, concepts: Set[str]) -> None:
        if analysis.primary_problem not in SIB_PROBLEM_DEFINITIONS:
            return
        problem_def = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem]
        for i, solution in enumerate(problem_def.solution_directions):
            solution_id = f"solution_{analysis.primary_problem.value}_{i}"
            G.add_node(
                solution_id, concept_type="solution", definition=solution,
                relevance=0.6, is_matched=False, is_inferred=True,
            )
            for concept in list(concepts)[:5]:
                if concept in G.nodes:
                    G.add_edge(solution_id, concept, rel_type="ENABLES", confidence=0.5,
                             weight=0.5, color="#90EE90", width=1.0, style="dashed")


# ============================================================================
# QUERY-ALIGNED VISUALIZER (Complete Implementation)
# ============================================================================
class QueryAlignedVisualizer:
    """Creates visualizations aligned with query focus."""

    TYPE_COLORS = {
        "material": "#FF6B6B",
        "process": "#4ECDC4",
        "property": "#45B7D1",
        "phenomenon": "#96CEB4",
        "method": "#FFEAA7",
        "parameter": "#DDA0DD",
        "model": "#F7DC6F",
        "solution": "#90EE90",
        "general": "#B0B0B0",
    }

    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology

    def create_pyvis_graph(self, G: nx.Graph, analysis: QueryAnalysisResult) -> str:
        if not PYVIS_AVAILABLE:
            return "<p>pyvis not installed. Install with: pip install pyvis</p>"

        net = Network(height="700px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.01,
                    "springLength": 150,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150}
            }
        }
        """)

        for node_id, node_data in G.nodes(data=True):
            relevance = node_data.get("relevance", 0.5)
            size = 15 + relevance * 35
            color = self.TYPE_COLORS.get(node_data.get("concept_type", "general"), "#B0B0B0")

            if node_data.get("is_matched"):
                border_width, border_color = 4, "#FFD700"
            elif node_data.get("is_inferred"):
                border_width, border_color = 2, "#87CEEB"
            else:
                border_width, border_color = 1, color

            title = f"<b>{node_id.replace('_', ' ').title()}</b>"
            if node_data.get("definition"):
                title += f"<br><i>{node_data['definition'][:100]}...</i>"
            title += f"<br>Relevance: {relevance:.2f}"

            net.add_node(node_id, label=node_id.replace("_", " ").title(), size=size,
                        color=color, border_width=border_width, border_color=border_color, title=title)

        for source, target, edge_data in G.edges(data=True):
            net.add_edge(source, target, color=edge_data.get("color", "#888888"),
                        width=edge_data.get("width", 1.0), style=edge_data.get("style", "solid"),
                        title=f"{edge_data.get('rel_type', '')}", arrows="to")

        return net.generate_html()

    def create_matplotlib_graph(self, G: nx.Graph, analysis: QueryAnalysisResult) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        node_colors = [self.TYPE_COLORS.get(G.nodes[n].get("concept_type", "general"), "#B0B0B0")
                      for n in G.nodes()]
        node_sizes = [200 + G.nodes[n].get("relevance", 0.5) * 500 for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)

        edge_colors = [G[u][v].get("color", "#888888") for u, v in G.edges()]
        edge_widths = [G[u][v].get("width", 1.0) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                              alpha=0.6, arrows=True, arrowsize=15, ax=ax)

        labels = {n: n.replace("_", "\n").title() for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white", ax=ax)

        ax.set_title(f"Concept Graph: {analysis.primary_problem.value.replace('_', ' ').title()}",
                    color="white", fontsize=16, pad=20)

        legend_patches = [mpatches.Patch(color=c, label=t.title()) for t, c in self.TYPE_COLORS.items()]
        ax.legend(handles=legend_patches, loc="upper left", facecolor="#2a2a4e", edgecolor="white",
                 labelcolor="white", fontsize=8)

        ax.axis("off")
        plt.tight_layout()
        return fig

    def create_problem_summary(self, analysis: QueryAnalysisResult) -> Dict[str, Any]:
        if analysis.primary_problem not in SIB_PROBLEM_DEFINITIONS:
            return {"error": "Unknown problem"}
        problem_def = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem]
        return {
            "problem_title": problem_def.title,
            "scientific_description": problem_def.scientific_description,
            "root_cause": problem_def.root_cause,
            "key_concepts": problem_def.key_concepts,
            "solution_directions": problem_def.solution_directions,
            "typical_quantities": problem_def.typical_quantities,
            "confidence": analysis.problem_confidences.get(analysis.primary_problem, 0),
            "matched_concepts": analysis.matched_concepts,
            "inferred_concepts": analysis.inferred_concepts,
            "emphasis_direction": analysis.emphasis_direction,
            "reasoning": analysis.reasoning,
        }

    def create_causal_chain_diagram(self, analysis: QueryAnalysisResult) -> plt.Figure:
        """Create a causal chain diagram for the identified problem."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        if analysis.primary_problem not in SIB_PROBLEM_DEFINITIONS:
            ax.text(0.5, 0.5, "No specific problem identified", ha="center", va="center",
                   color="white", fontsize=14)
            return fig

        problem_def = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem]

        # Build causal chain
        chain = []
        for src, rel, tgt in problem_def.key_relationships:
            if src in self.ontology.concepts and tgt in self.ontology.concepts:
                chain.append((src, rel, tgt))

        if not chain:
            ax.text(0.5, 0.5, "No causal chain available", ha="center", va="center",
                   color="white", fontsize=14)
            return fig

        # Layout
        n_nodes = len(set([n for c in chain for n in c[:2]] + [c[2] for c in chain]))
        pos = {}
        y_positions = np.linspace(0.9, 0.1, min(n_nodes, 8))

        node_list = list(dict.fromkeys([c[0] for c in chain] + [c[2] for c in chain]))[:8]
        for i, node in enumerate(node_list):
            pos[node] = (0.5 + 0.2 * np.sin(i * 0.8), y_positions[i])

        # Draw edges
        for src, rel, tgt in chain[:10]:
            if src in pos and tgt in pos:
                ax.annotate("", xy=pos[tgt], xytext=pos[src],
                           arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=2))
                mid_x = (pos[src][0] + pos[tgt][0]) / 2
                mid_y = (pos[src][1] + pos[tgt][1]) / 2
                ax.text(mid_x + 0.05, mid_y, rel, fontsize=8, color="#FFEAA7", style="italic")

        # Draw nodes
        for node, (x, y) in pos.items():
            color = self.TYPE_COLORS.get(
                self.ontology.get_concept_type(node).value if node in self.ontology.concepts else "general",
                "#B0B0B0"
            )
            circle = plt.Circle((x, y), 0.04, color=color, ec="white", lw=2)
            ax.add_patch(circle)
            ax.text(x, y, node.replace("_", "\n")[:15], ha="center", va="center",
                   fontsize=7, color="white", weight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Causal Chain: {problem_def.title}", color="white", fontsize=14, pad=20)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def create_quantities_bar_chart(self, analysis: QueryAnalysisResult) -> plt.Figure:
        """Create bar chart of typical quantities for the problem."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        if analysis.primary_problem not in SIB_PROBLEM_DEFINITIONS:
            return fig

        quantities = SIB_PROBLEM_DEFINITIONS[analysis.primary_problem].typical_quantities
        if not quantities:
            ax.text(0.5, 0.5, "No quantitative data available", ha="center", va="center",
                   color="white", fontsize=14)
            return fig

        names = list(quantities.keys())
        mins = [quantities[n][0] for n in names]
        maxs = [quantities[n][1] for n in names]
        units = [quantities[n][2] for n in names]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, mins, width, label='Min', color='#4ECDC4', alpha=0.8)
        bars2 = ax.bar(x + width/2, maxs, width, label='Max', color='#FF6B6B', alpha=0.8)

        ax.set_ylabel(f'Value', color="white", fontsize=12)
        ax.set_title(f'Typical Quantitative Ranges: {analysis.primary_problem.value.replace("_", " ").title()}',
                    color="white", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n({u})" for n, u in zip(names, units)], rotation=45, ha="right",
                          color="white", fontsize=9)
        ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
        ax.tick_params(colors="white")
        ax.spines['bottom'].set_color("white")
        ax.spines['left'].set_color("white")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE QUERIES
# ============================================================================
EXAMPLE_QUERIES_BY_PROBLEM = {
    SIBCoreProblem.ANODE_BOTTLENECK: [
        "Why can't sodium intercalate into graphite like lithium does?",
        "What is the sodium storage mechanism in hard carbon anodes?",
        "How can we improve the initial Coulombic efficiency of hard carbon?",
        "Compare hard carbon vs. alloying anodes for SIBs",
    ],
    SIBCoreProblem.CATHODE_INSTABILITY: [
        "What causes the P2 to O2 phase transition in NaₓMnO₂ cathodes?",
        "How does elemental doping stabilize layered oxide cathodes?",
        "Compare layered vs. polyanionic cathodes for structural stability",
    ],
    SIBCoreProblem.SEI_CHEMISTRY: [
        "Why is the SEI in SIBs less stable than in LIBs?",
        "How do concentrated electrolytes improve SEI formation?",
        "What role do fluorinated additives play in SEI stabilization?",
    ],
    SIBCoreProblem.SOLID_STATE_INTERFACE: [
        "What causes void formation at the solid electrolyte interface?",
        "How can quasi-solid electrolytes improve solid-state SIBs?",
        "Compare NASICON vs. sulfide solid electrolytes",
    ],
    SIBCoreProblem.LOW_ENERGY_DENSITY: [
        "What is the theoretical energy density limit for SIBs vs LIBs?",
        "How can high-voltage cathodes push SIB energy density?",
        "Can SIBs achieve energy density competitive with LIBs for EVs?",
    ],
    SIBCoreProblem.MOISTURE_MANUFACTURING: [
        "Why are sodium cathodes more moisture-sensitive than lithium?",
        "How does surface alkalinity affect slurry coating quality?",
        "What surface washing treatments stabilize sodium cathodes?",
    ],
}


# ============================================================================
# COMPLETE STREAMLIT UI
# ============================================================================
def main():
    """Main Streamlit application."""
    
    # Initialize ontology
    @st.cache_resource
    def get_ontology():
        return DomainOntology()
    
    ontology = get_ontology()
    
    # Initialize components
    analyzer = SIBQueryAnalyzer(ontology=ontology, mode="rule_based")
    builder = DirectedConceptGraphBuilder(ontology)
    visualizer = QueryAlignedVisualizer(ontology)
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .problem-card {
        background: linear-gradient(135deg, #2a2a4e 0%, #1a1a2e 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 10px 0;
    }
    .metric-card {
        background: #2a2a4e;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
    <h1 style="color: white; margin: 0;">🔋 SIB LLM-Directed Concept Graph</h1>
    <p style="color: #B0B0B0; margin: 5px 0 0 0;">Intelligent query analysis for Sodium-Ion Battery research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### LLM Mode")
        llm_mode = st.selectbox(
            "Select LLM Mode:",
            ["rule_based", "local", "api"],
            format_func=lambda x: {"rule_based": "Rule-Based (No LLM)", 
                                   "local": "Local Model (GPT-2/Qwen)",
                                   "api": "OpenAI API"}[x]
        )
        
        if llm_mode == "api":
            api_key = st.text_input("API Key:", type="password")
            api_model = st.text_input("Model:", value="gpt-3.5-turbo")
        elif llm_mode == "local":
            model_name = st.text_input("Model Name:", value="gpt2")
        
        st.markdown("---")
        st.markdown("### Example Queries")
        problem_select = st.selectbox(
            "Problem Category:",
            [""] + [p.value for p in SIBCoreProblem if p in EXAMPLE_QUERIES_BY_PROBLEM],
            format_func=lambda x: SIB_PROBLEM_DEFINITIONS[SIBCoreProblem(x)].title if x else "Select..."
        )
        
        if problem_select:
            problem = SIBCoreProblem(problem_select)
            examples = EXAMPLE_QUERIES_BY_PROBLEM.get(problem, [])
            example_select = st.selectbox("Select Example:", [""] + examples)
            if example_select:
                st.session_state["query"] = example_select
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system analyzes SIB research queries and:
        1. Classifies into 6 core problems
        2. Extracts ontology concepts
        3. Builds focused concept graphs
        4. Generates aligned visualizations
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "🔬 Enter your SIB research query:",
            value=st.session_state.get("query", ""),
            placeholder="e.g., Why can't sodium intercalate into graphite like lithium does?",
            height=100,
            key="query_input"
        )
    
    with col2:
        analyze_btn = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
        if clear_btn:
            st.session_state.clear()
            st.rerun()
    
    # Analysis results
    if analyze_btn and query:
        with st.spinner("Analyzing query..."):
            # Update analyzer mode
            if llm_mode == "api":
                analyzer = SIBQueryAnalyzer(ontology, mode="api", api_key=api_key, api_model=api_model)
            elif llm_mode == "local":
                analyzer = SIBQueryAnalyzer(ontology, mode="local", model_name=model_name)
            else:
                analyzer = SIBQueryAnalyzer(ontology, mode="rule_based")
            
            analysis = analyzer.analyze_query(query)
            G = builder.build_focused_subgraph(analysis)
            
            # Display analysis
            st.markdown("---")
            display_analysis_results(analysis)
            
            # Tabs for visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Concept Graph",
                "📋 Problem Summary",
                "🔗 Causal Chain",
                "📈 Quantities",
                "🎯 Concept Details"
            ])
            
            with tab1:
                st.subheader("Interactive Concept Graph")
                if PYVIS_AVAILABLE:
                    html_output = visualizer.create_pyvis_graph(G, analysis)
                    st.components.v1.html(html_output, height=750)
                else:
                    fig = visualizer.create_matplotlib_graph(G, analysis)
                    st.pyplot(fig)
            
            with tab2:
                st.subheader("Problem Summary")
                summary = visualizer.create_problem_summary(analysis)
                display_problem_summary(summary)
            
            with tab3:
                st.subheader("Causal Chain Diagram")
                fig = visualizer.create_causal_chain_diagram(analysis)
                st.pyplot(fig)
            
            with tab4:
                st.subheader("Typical Quantitative Ranges")
                fig = visualizer.create_quantities_bar_chart(analysis)
                st.pyplot(fig)
            
            with tab5:
                st.subheader("Concept Details")
                display_concept_details(analysis, G)
    
    elif not query:
        st.markdown("---")
        st.info("👈 Enter a query or select an example from the sidebar to begin analysis.")
        
        # Show problem overview
        st.markdown("### The Six Core SIB Problems")
        cols = st.columns(3)
        problems = list(SIB_PROBLEM_DEFINITIONS.keys())[:6]
        for i, problem in enumerate(problems):
            with cols[i % 3]:
                problem_def = SIB_PROBLEM_DEFINITIONS[problem]
                st.markdown(f"""
                <div class="problem-card">
                <h4 style="color: #FF6B6B; margin: 0 0 5px 0;">{i+1}. {problem_def.title.split(':')[0]}</h4>
                <p style="color: #B0B0B0; font-size: 12px; margin: 0;">{problem_def.scientific_description[:150]}...</p>
                </div>
                """, unsafe_allow_html=True)


def display_analysis_results(analysis: QueryAnalysisResult):
    """Display query analysis results."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Primary Problem")
        problem = analysis.primary_problem
        if problem in SIB_PROBLEM_DEFINITIONS:
            st.info(f"**{SIB_PROBLEM_DEFINITIONS[problem].title}**")
            st.metric("Confidence", f"{analysis.problem_confidences.get(problem, 0):.1%}")
    
    with col2:
        st.markdown("### 📊 Emphasis")
        st.metric("Direction", analysis.emphasis_direction.replace("_", " ").title())
        if analysis.emphasis_keywords:
            st.markdown(f"**Keywords:** {', '.join(analysis.emphasis_keywords)}")
    
    with col3:
        st.markdown("### 🔗 Concepts")
        st.metric("Matched", len(analysis.matched_concepts))
        st.metric("Inferred", len(analysis.inferred_concepts))
    
    if analysis.secondary_problems:
        st.markdown("### Related Problems")
        cols = st.columns(len(analysis.secondary_problems))
        for i, sec_prob in enumerate(analysis.secondary_problems):
            with cols[i]:
                if sec_prob in SIB_PROBLEM_DEFINITIONS:
                    conf = analysis.problem_confidences.get(sec_prob, 0)
                    st.markdown(f"**{SIB_PROBLEM_DEFINITIONS[sec_prob].title[:40]}** ({conf:.0%})")
    
    st.markdown("### 💡 Reasoning")
    st.info(analysis.reasoning)


def display_problem_summary(summary: Dict[str, Any]):
    """Display problem summary panel."""
    st.markdown(f"## {summary.get('problem_title', 'Unknown')}")
    st.write(summary.get('scientific_description', ''))
    
    with st.expander("🔬 Root Cause", expanded=False):
        st.warning(summary.get('root_cause', ''))
    
    with st.expander("🔑 Key Concepts", expanded=True):
        cols = st.columns(3)
        for i, concept in enumerate(summary.get('key_concepts', [])):
            with cols[i % 3]:
                st.markdown(f"- `{concept}`")
    
    with st.expander("💡 Solution Directions", expanded=True):
        for i, solution in enumerate(summary.get('solution_directions', []), 1):
            st.markdown(f"{i}. {solution}")
    
    with st.expander("📊 Typical Quantities", expanded=False):
        for name, (min_val, max_val, unit) in summary.get('typical_quantities', {}).items():
            st.markdown(f"- **{name}**: {min_val} - {max_val} {unit}")


def display_concept_details(analysis: QueryAnalysisResult, G: nx.Graph):
    """Display detailed concept information."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Matched Concepts (from query)")
        for concept in analysis.matched_concepts:
            if concept in G.nodes:
                node_data = G.nodes[concept]
                relevance = node_data.get('relevance', 0)
                st.markdown(f"**{concept}** (relevance: {relevance:.2f})")
                if node_data.get('definition'):
                    st.caption(node_data['definition'][:100] + "...")
    
    with col2:
        st.markdown("### 🔍 Inferred Concepts (contextual)")
        for concept in analysis.inferred_concepts:
            if concept in G.nodes:
                node_data = G.nodes[concept]
                relevance = node_data.get('relevance', 0)
                st.markdown(f"*{concept}* (relevance: {relevance:.2f})")
                if node_data.get('definition'):
                    st.caption(node_data['definition'][:100] + "...")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
